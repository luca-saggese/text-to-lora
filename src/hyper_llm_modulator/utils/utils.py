import logging
from math import sqrt
from typing import Literal

import torch
from tqdm import tqdm
import wandb

logger = logging.getLogger()


def embed_texts(texts, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device, batch_size=None):
    formatted_descs = list(map(task_desc_format_fn, texts))
    logger.info(f"embed_texts: Processing {len(texts)} texts, device={device}")
    logger.info(f"embed_texts: First formatted desc: {formatted_descs[0][:100] if formatted_descs else 'N/A'}")
    
    tokenized_ds_descs = emb_tokenizer(
        formatted_descs,
        truncation=True,
        padding=True,
        max_length=2**13,
        return_tensors="pt",
    )
    
    logger.info(f"embed_texts: Tokenizer output keys: {list(tokenized_ds_descs.keys())}")
    for key, val in tokenized_ds_descs.items():
        if val is not None:
            logger.info(f"  {key}: shape={val.shape}, dtype={val.dtype}, min={val.min().item()}, max={val.max().item()}")

    # Strict validation and alignment
    input_ids = tokenized_ds_descs.get("input_ids")
    input_embeddings = emb_model.get_input_embeddings() if hasattr(emb_model, "get_input_embeddings") else None
    model_vocab_size = input_embeddings.num_embeddings if input_embeddings is not None else None
    tokenizer_vocab_size = emb_tokenizer.vocab_size if hasattr(emb_tokenizer, "vocab_size") else None
    
    logger.info(f"embed_texts: Vocab check - tokenizer_vocab={tokenizer_vocab_size}, model_vocab={model_vocab_size}")
    
    # Ensure input_ids is proper dtype
    if input_ids is not None:
        logger.info(f"embed_texts: input_ids before conversion: dtype={input_ids.dtype}")
        tokenized_ds_descs["input_ids"] = input_ids.to(torch.long)
        input_ids = tokenized_ds_descs["input_ids"]
        logger.info(f"embed_texts: input_ids after conversion: dtype={input_ids.dtype}")
    
    # Validate and fix out-of-vocab tokens
    if input_ids is not None and model_vocab_size is not None:
        invalid_mask = (input_ids < 0) | (input_ids >= model_vocab_size)
        if invalid_mask.any():
            invalid_count = int(invalid_mask.sum().item())
            invalid_max = int(input_ids[invalid_mask].max().item())
            logger.warning(f"embed_texts: Found {invalid_count} out-of-vocab tokens, max={invalid_max}")
            replacement_id = emb_tokenizer.unk_token_id
            if replacement_id is None or replacement_id >= model_vocab_size:
                replacement_id = emb_tokenizer.pad_token_id
            if replacement_id is None or replacement_id >= model_vocab_size:
                replacement_id = emb_tokenizer.eos_token_id
            if replacement_id is None or replacement_id >= model_vocab_size:
                replacement_id = 0
            
            logger.warning(f"embed_texts: Replacing with token_id={replacement_id}")
            tokenized_ds_descs["input_ids"][invalid_mask] = replacement_id
    
    # Ensure attention_mask exists and is proper dtype
    if "attention_mask" not in tokenized_ds_descs or tokenized_ds_descs["attention_mask"] is None:
        if input_ids is not None:
            tokenized_ds_descs["attention_mask"] = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
            logger.info("embed_texts: Created attention_mask as bool")
    else:
        original_dtype = tokenized_ds_descs["attention_mask"].dtype
        # Keep attention_mask as bool (the proper type for attention masks)
        tokenized_ds_descs["attention_mask"] = tokenized_ds_descs["attention_mask"].to(torch.bool)
        logger.info(f"embed_texts: Converted attention_mask from {original_dtype} to torch.bool")
    
    # CRITICAL: Remove position_ids if present - we will generate them ourselves in _embed_tokens_single_batch
    if "position_ids" in tokenized_ds_descs:
        logger.info(f"embed_texts: REMOVING position_ids from tokenizer output (shape={tokenized_ds_descs['position_ids'].shape})")
        del tokenized_ds_descs["position_ids"]
    
    # CRITICAL: Remove token_type_ids - many embedding models don't use it and it can corrupt position handling
    if "token_type_ids" in tokenized_ds_descs:
        logger.info(f"embed_texts: REMOVING token_type_ids (shape={tokenized_ds_descs['token_type_ids'].shape})")
        del tokenized_ds_descs["token_type_ids"]
    
    # Remove any other unexpected keys
    # Keep only: input_ids, attention_mask (position_ids will be generated in _embed_tokens_single_batch)
    valid_keys = {"input_ids", "attention_mask"}
    for key in list(tokenized_ds_descs.keys()):
        if key not in valid_keys:
            logger.info(f"embed_texts: REMOVING unexpected key {key}")
            del tokenized_ds_descs[key]

    return embed_tokens(tokenized_ds_descs, emb_model, pooling_fn, device, batch_size)


def embed_tokens(tokenized_texts, emb_model, pooling_fn, device, batch_size=None):
    # Ensure proper dtype ONLY for input_ids (NOT attention_mask - keep it as bool)
    logger.info(f"embed_tokens: Input keys: {list(tokenized_texts.keys())}")
    if "input_ids" in tokenized_texts and tokenized_texts["input_ids"] is not None:
        original_dtype = tokenized_texts["input_ids"].dtype
        logger.info(f"embed_tokens: input_ids before to(long): dtype={original_dtype}, shape={tokenized_texts['input_ids'].shape}")
        tokenized_texts["input_ids"] = tokenized_texts["input_ids"].to(torch.long)
        logger.info(f"embed_tokens: input_ids after to(long): dtype={tokenized_texts['input_ids'].dtype}")
    
    if batch_size is None:
        # Process all at once if no batch size specified
        logger.info("embed_tokens: Processing all texts at once (no batch size)")
        tokenized_texts = {k: v.to(device) if v is not None else None for k, v in tokenized_texts.items()}
        logger.info(f"embed_tokens: Data moved to device {device}")
        for k, v in tokenized_texts.items():
            if v is not None:
                logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
        return _embed_tokens_single_batch(tokenized_texts, emb_model, pooling_fn)

    # Process in batches
    n_samples = tokenized_texts["input_ids"].shape[0]
    logger.info(f"embed_tokens: Processing {n_samples} samples in batches of {batch_size}")
    embeddings = []

    for start_idx in tqdm(range(0, n_samples, batch_size), total=n_samples // batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = {k: v[start_idx:end_idx].to(device) if v is not None else None for k, v in tokenized_texts.items()}
        batch_embeddings = _embed_tokens_single_batch(batch, emb_model, pooling_fn)
        embeddings.append(batch_embeddings)

    return torch.cat(embeddings, dim=0)


def _embed_tokens_single_batch(tokenized_texts, emb_model, pooling_fn):
    # Validate input tensors before model forward
    logger.info("_embed_tokens_single_batch: Starting forward pass")
    logger.info(f"_embed_tokens_single_batch: Input keys: {list(tokenized_texts.keys())}")
    
    for key, val in tokenized_texts.items():
        if val is not None:
            logger.info(f"  {key}: shape={val.shape}, dtype={val.dtype}, device={val.device}")
            if key == "input_ids":
                logger.info(f"    {key} sample values (first 10): {val.flatten()[:10].tolist()}")
            if key == "attention_mask":
                logger.info(f"    {key} sample values (first 10): {val.flatten()[:10].tolist()}")
    
    if "input_ids" in tokenized_texts:
        input_ids = tokenized_texts["input_ids"]
        if input_ids is not None:
            min_id = input_ids.min().item()
            max_id = input_ids.max().item()
            logger.info(f"_embed_tokens_single_batch: input_ids range: min={min_id}, max={max_id}")
            if min_id < 0 or max_id >= 128000:
                logger.warning(
                    "Suspicious input_ids range: min=%d, max=%d. This may cause indexing errors.",
                    min_id,
                    max_id,
                )
    
    if "position_ids" in tokenized_texts and tokenized_texts["position_ids"] is not None:
        pos_ids = tokenized_texts["position_ids"]
        logger.warning(f"CRITICAL: position_ids found! shape={pos_ids.shape}, dtype={pos_ids.dtype}, min={pos_ids.min().item()}, max={pos_ids.max().item()}")
    
    # CRITICAL FIX: Generate position_ids ourselves instead of letting Alibaba model do it
    # This prevents the model from generating corrupted position IDs internally
    if "input_ids" in tokenized_texts:
        batch_size, seq_len = tokenized_texts["input_ids"].shape
        device = tokenized_texts["input_ids"].device
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        tokenized_texts["position_ids"] = position_ids
        logger.info(f"_embed_tokens_single_batch: Generated position_ids: shape={position_ids.shape}, dtype={position_ids.dtype}, values={position_ids.flatten()[:10].tolist()}")
    
    # Ensure model is in eval mode and use no_grad for inference
    was_training = emb_model.training
    emb_model.eval()
    logger.info(f"_embed_tokens_single_batch: Model training mode before={was_training}, now={emb_model.training}")
    
    try:
        logger.info("_embed_tokens_single_batch: Calling emb_model.forward() WITH generated position_ids and torch.no_grad()")
        with torch.no_grad():
            outputs = emb_model(**tokenized_texts)
        logger.info(f"_embed_tokens_single_batch: Forward pass succeeded")
        logger.info(f"_embed_tokens_single_batch: Output type: {type(outputs)}, keys: {list(outputs.keys()) if hasattr(outputs, 'keys') else 'N/A'}")
    except IndexError as e:
        logger.error(
            "IndexError during embedding model forward pass. "
            "This often indicates token ID or position ID mismatch. "
            "Input summary: %s",
            {k: f"shape={v.shape}, dtype={v.dtype}" if v is not None else "None" for k, v in tokenized_texts.items()},
        )
        logger.error(f"Full error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during embedding forward: {type(e).__name__}: {str(e)}")
        raise
    finally:
        # Restore original training mode
        if was_training:
            emb_model.train()
    
    logger.info("_embed_tokens_single_batch: Computing pooling")
    task_embs = pooling_fn(outputs, tokenized_texts["attention_mask"]).to(torch.float32)
    logger.info(f"_embed_tokens_single_batch: task_embs shape={task_embs.shape}, dtype={task_embs.dtype}")
    
    normalized = torch.nn.functional.normalize(task_embs) * sqrt(task_embs.shape[-1])
    logger.info(f"_embed_tokens_single_batch: final embeddings shape={normalized.shape}")
    return normalized


def get_inp_tokenize_fn(
    tokenizer,
    sft_mode: Literal["causal_lm", "completion"],
    is_intx_model: bool,
    inp_max_len: int,
):
    def tokenize_causal_lm(examples):
        # a dict with keys: ["input_ids", "attention_mask"]
        tokenized_seq = tokenizer(
            examples["text"],
            # apply_chat_template should already add all the special tokens
            add_special_tokens=True if not is_intx_model else False,
            truncation=True,
            padding=False,
            max_length=inp_max_len,
        )
        tokenized_seq["labels"] = tokenized_seq["input_ids"]
        return tokenized_seq

    # NOTE: we're not considering multi-turn sft
    # this fn is used to mask out the loss from the prompt
    # and train only on the response
    # see # see https://github.com/huggingface/trl/issues/632#issuecomment-1972630547
    # https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb
    # for more advanced multi-turn training
    def tokenize_prompt_completion(examples):
        # a dict with keys: ["input_ids", "attention_mask"]
        # we can also access seqeunce_ids to differentiate between prompt and response
        tokenized_seq = tokenizer(
            examples["prompt"],
            examples["response"],
            add_special_tokens=False,
            truncation=True,
            padding=False,
            # apply to prompt and response separately
            # i.e., we can get the max sequence length of 2 x inp_max_len
            max_length=inp_max_len,
        )

        tokenized_seq["labels"] = [None] * len(tokenized_seq["input_ids"])
        input_ids = tokenized_seq["input_ids"]
        attention_mask = tokenized_seq["attention_mask"]
        labels = tokenized_seq["labels"]
        for i in range(len(tokenized_seq["input_ids"])):
            if not is_intx_model:
                # manually add bos and eos tokens
                input_ids[i] = [tokenizer.bos_token_id] + input_ids[i] + [tokenizer.eos_token_id]
                attention_mask[i] = [1] + attention_mask[i] + [1]
                sequence_ids = [0] + tokenized_seq.sequence_ids(i) + [1]
            else:
                sequence_ids = tokenized_seq.sequence_ids(i)
            labels[i] = [-100 if sequence_id == 0 else label for sequence_id, label in zip(sequence_ids, input_ids[i])]
        return tokenized_seq

    tokenize_function = tokenize_causal_lm if sft_mode == "causal_lm" else tokenize_prompt_completion
    return tokenize_function


def log_scalar(metric_name, val, curstep):
    if wandb.run is not None:
        wandb.log({metric_name: val}, step=curstep)
    logger.info(f"{metric_name}: {val:.4f}")
