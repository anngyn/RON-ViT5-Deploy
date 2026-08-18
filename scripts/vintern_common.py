"""Shared helpers for the Vintern-1B integration (Method A: text-only).

Vintern-1B is a multimodal ViT-MLP-LLM model (InternViT-300M + Qwen2-0.5B). Method A
uses it as a *text-only* QA model: we ignore the vision tower and LoRA-fine-tune the
embedded Qwen2 language model on "question + OCR context -> answer", injecting the same
OCR noise the seq2seq pipeline uses. This keeps the OCR-noise-robustness comparison and
adds a decoder-only-LLM data point next to the ViT5/mT5/BARTpho encoder-decoders.

This module is self-contained: it reuses only src/data (load_data), src/noise
(OCRNoiseGenerator) and src/evaluate (compute_anls). It does NOT touch src/train.py,
src/dataset.py or any existing script.

Requirements (install once on the runner):
    pip install "transformers>=4.37" peft accelerate timm einops sentencepiece

NOTE: this code has not been executed in this environment (no GPU here). The Vintern
remote code exposes the language model as ``model.language_model``; verify that on first
run (the loader raises a clear error otherwise).
"""
import os
import sys
import types
import importlib.util

# --------------------------------------------------------------------------- #
# flash_attn stub (must run before any Vintern remote code is imported)
# --------------------------------------------------------------------------- #
# Vintern's remote modeling files (modeling_intern_vit.py / modeling_internvl_chat.py)
# reference `flash_attn` at module top level. transformers' `check_imports` statically
# scans those files and refuses to load the model when flash_attn is absent -- even
# though Method A passes use_flash_attn=False and never launches a flash-attention
# kernel. Building flash-attn from source is slow and brittle (notably on Kaggle T4),
# so we register a lightweight stub instead. It satisfies the static import check; the
# dummy callables raise only if flash attention is actually invoked, which cannot happen
# while use_flash_attn=False (the model selects the eager attention path).
if "flash_attn" not in sys.modules:
    try:
        import flash_attn  # noqa: F401  (real package present -> use it)
    except ImportError:
        def _flash_attn_unavailable(*args, **kwargs):  # pragma: no cover
            raise RuntimeError(
                "flash_attn is a stub (not installed). This path is unreachable "
                "while use_flash_attn=False."
            )

        _fa_stub = types.ModuleType("flash_attn")
        _fa_stub.__version__ = "0.0.0"
        # A valid __spec__ is required: transformers probes flash-attn with
        # importlib.util.find_spec(), which raises "flash_attn.__spec__ is None"
        # on a bare types.ModuleType. With a spec present, find_spec() succeeds,
        # then importlib.metadata.version() finds no distribution -> transformers
        # concludes flash-attn is not installed and selects the eager path.
        _fa_stub.__spec__ = importlib.util.spec_from_loader("flash_attn", loader=None)
        for _fa_name in (
            "flash_attn_func",
            "flash_attn_varlen_func",
            "flash_attn_qkvpacked_func",
            "flash_attn_varlen_qkvpacked_func",
        ):
            setattr(_fa_stub, _fa_name, _flash_attn_unavailable)

        _fa_bert_padding = types.ModuleType("flash_attn.bert_padding")
        _fa_bert_padding.__spec__ = importlib.util.spec_from_loader(
            "flash_attn.bert_padding", loader=None
        )
        for _fa_name in ("index_first_axis", "pad_input", "unpad_input"):
            setattr(_fa_bert_padding, _fa_name, _flash_attn_unavailable)

        sys.modules["flash_attn"] = _fa_stub
        sys.modules["flash_attn.bert_padding"] = _fa_bert_padding

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import load_data  # noqa: E402  (reused, unchanged)
from src.noise import OCRNoiseGenerator  # noqa: E402
from src.evaluate import compute_anls  # noqa: E402

# Prompt shown to the model. Kept short; the answer must be terse for ANLS.
SYSTEM_PROMPT = "Bạn là trợ lý đọc hóa đơn. Chỉ trả lời ngắn gọn, đúng trọng tâm câu hỏi."
USER_TEMPLATE = "Nội dung hóa đơn (OCR):\n{context}\n\nCâu hỏi: {question}\nTrả lời:"

IGNORE_INDEX = -100


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_vintern(model_name, device, dtype=torch.float16):
    """Load Vintern and return (wrapper_model, tokenizer, language_model).

    ``language_model`` is the embedded Qwen2 causal LM that Method A fine-tunes /
    generates from. The vision tower is never called.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Vintern-1B-v2's InternVLChatModel.__init__ signature is (config, vision_model,
    # language_model) -- it does NOT accept a `use_flash_attn` kwarg (unlike upstream
    # InternVL2). It builds its Qwen2 language model directly from `config.llm_config`,
    # whose attention class is picked from `_attn_implementation`. Method A only ever
    # runs the Qwen2 LLM, so we force eager attention there; combined with the flash_attn
    # stub above this guarantees no flash-attention kernel is ever launched.
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "llm_config"):
        config.llm_config._attn_implementation = "eager"
    model = AutoModel.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise AttributeError(
            "Vintern wrapper has no `.language_model`. Inspect the loaded model's "
            "attributes and adjust load_vintern() to point at the Qwen2 causal LM."
        )
    language_model.to(device)
    return model, tokenizer, language_model


def apply_lora(language_model, config):
    """Wrap the language model with a LoRA adapter (PEFT). Returns the PEFT model."""
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("lora_targets", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type="CAUSAL_LM",
        bias="none",
    )
    peft_model = get_peft_model(language_model, lora)
    peft_model.print_trainable_parameters()
    return peft_model


# --------------------------------------------------------------------------- #
# Prompt / tokenization
# --------------------------------------------------------------------------- #
def build_context(ocr_texts, noise_gen=None, noise_type=None, level=2):
    """Join OCR tokens into a context string, optionally applying OCR noise."""
    context = " ".join(str(t) for t in ocr_texts)
    if noise_gen is not None and noise_type is not None:
        context = noise_gen.apply_noise(context, noise_type, level)
    return context


def _prompt_text(tokenizer, question, context):
    """Render the chat prompt up to the assistant turn (Qwen2 chat template)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=str(question).strip())},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_train_example(tokenizer, question, context, answer, max_length):
    """Build (input_ids, labels) for causal SFT with answer-only loss.

    Prompt tokens are masked with IGNORE_INDEX so the loss is computed only on the
    answer + eos. If the sequence is too long, the *prompt* is truncated from the left
    to always keep the answer.
    """
    prompt_ids = tokenizer(_prompt_text(tokenizer, question, context), add_special_tokens=False).input_ids
    answer_ids = tokenizer(str(answer).strip(), add_special_tokens=False).input_ids + [tokenizer.eos_token_id]

    keep_prompt = max(0, max_length - len(answer_ids))
    if len(prompt_ids) > keep_prompt:
        prompt_ids = prompt_ids[-keep_prompt:]  # keep the end (question) nearest the answer

    input_ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
    return input_ids[:max_length], labels[:max_length]


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class VinternQADataset(Dataset):
    """Causal-SFT dataset. mode = 'clean' | 'noisy' | 'paired'.

    - clean : context is the raw OCR text.
    - noisy : each sample is randomly corrupted (noisy augmentation).
    - paired: returns clean AND noisy tensors of the same sample (for consistency).
    """

    def __init__(self, qa_df, ocr_df, tokenizer, mode="clean", noise_gen=None,
                 noise_types=("mixed_noise",), noise_levels=(2,), max_length=1024):
        import random
        import pandas as pd

        self.tokenizer = tokenizer
        self.mode = mode
        self.noise_gen = noise_gen
        self.noise_types = list(noise_types)
        self.noise_levels = list(noise_levels)
        self.max_length = max_length
        self._random = random.Random(42)

        merged = pd.merge(qa_df, ocr_df[["image_id", "texts"]], on="image_id", how="inner")
        self.samples = merged[["question", "texts", "answer"]].to_dict("records")

    def __len__(self):
        return len(self.samples)

    def _noisy_context(self, texts):
        ntype = self._random.choice(self.noise_types)
        level = self._random.choice(self.noise_levels)
        return build_context(texts, self.noise_gen, ntype, level)

    def __getitem__(self, idx):
        s = self.samples[idx]
        clean_ctx = build_context(s["texts"])

        if self.mode == "clean":
            ids, labels = tokenize_train_example(self.tokenizer, s["question"], clean_ctx, s["answer"], self.max_length)
            return {"input_ids": ids, "labels": labels}

        if self.mode == "noisy":
            ctx = self._noisy_context(s["texts"])
            ids, labels = tokenize_train_example(self.tokenizer, s["question"], ctx, s["answer"], self.max_length)
            return {"input_ids": ids, "labels": labels}

        # paired: clean + noisy of the same answer
        noisy_ctx = self._noisy_context(s["texts"])
        ci, cl = tokenize_train_example(self.tokenizer, s["question"], clean_ctx, s["answer"], self.max_length)
        ni, nl = tokenize_train_example(self.tokenizer, s["question"], noisy_ctx, s["answer"], self.max_length)
        return {"clean_input_ids": ci, "clean_labels": cl, "noisy_input_ids": ni, "noisy_labels": nl}


def _pad_batch(seqs, pad_value):
    max_len = max(len(s) for s in seqs)
    padded = [s + [pad_value] * (max_len - len(s)) for s in seqs]
    mask = [[1] * len(s) + [0] * (max_len - len(s)) for s in seqs]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def make_collate(tokenizer, mode):
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        if mode in ("clean", "noisy"):
            input_ids, attn = _pad_batch([b["input_ids"] for b in batch], pad_id)
            labels, _ = _pad_batch([b["labels"] for b in batch], IGNORE_INDEX)
            return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
        # paired
        out = {}
        for side in ("clean", "noisy"):
            ids, attn = _pad_batch([b[f"{side}_input_ids"] for b in batch], pad_id)
            labels, _ = _pad_batch([b[f"{side}_labels"] for b in batch], IGNORE_INDEX)
            out[f"{side}_input_ids"] = ids
            out[f"{side}_attention_mask"] = attn
            out[f"{side}_labels"] = labels
        return out

    return collate


# --------------------------------------------------------------------------- #
# Training loops (causal)
# --------------------------------------------------------------------------- #
def train_epoch_causal(model, loader, optimizer, device, scaler=None, grad_accum=1, epoch_num=0):
    """Standard causal-LM SFT epoch (baseline / noisy-aug flows).

    Uses autocast(fp16) + GradScaler when a scaler is provided. Pure-fp16 training
    (no autocast/scaler) overflows softmax/cross-entropy on sm_75 GPUs and drives the
    LoRA weights to NaN, which collapses ANLS to 0; the scaler skips inf/nan steps and
    keeps the trainable (fp32) parameters stable.
    """
    from tqdm import tqdm

    use_amp = scaler is not None and scaler.is_enabled()
    model.train()
    total = 0.0
    optimizer.zero_grad()
    bar = tqdm(loader, desc=f"Epoch {epoch_num}")
    for step, batch in enumerate(bar):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
        loss = out.loss / grad_accum
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % grad_accum == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        total += out.loss.item()
        bar.set_postfix({"loss": f"{out.loss.item():.4f}"})
    return total / max(1, len(loader))


def _masked_mean(hidden, labels):
    """Mean-pool last hidden state over answer positions (labels != IGNORE_INDEX)."""
    mask = (labels != IGNORE_INDEX).unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def train_epoch_causal_consistency(model, loader, optimizer, device, beta, scaler=None,
                                   grad_accum=1, epoch_num=0):
    """Consistency flow: CE_clean + CE_noisy + beta * (1 - cos) on the answer-region
    hidden states of the clean vs noisy pass (same answer). Uses autocast(fp16) +
    GradScaler when a scaler is provided (see train_epoch_causal for the rationale)."""
    from tqdm import tqdm

    use_amp = scaler is not None and scaler.is_enabled()
    model.train()
    total = 0.0
    optimizer.zero_grad()
    bar = tqdm(loader, desc=f"Epoch {epoch_num}")
    for step, batch in enumerate(bar):
        clean_labels = batch["clean_labels"].to(device)
        noisy_labels = batch["noisy_labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out_clean = model(
                input_ids=batch["clean_input_ids"].to(device),
                attention_mask=batch["clean_attention_mask"].to(device),
                labels=clean_labels,
                output_hidden_states=True,
            )
            out_noisy = model(
                input_ids=batch["noisy_input_ids"].to(device),
                attention_mask=batch["noisy_attention_mask"].to(device),
                labels=noisy_labels,
                output_hidden_states=True,
            )
            h_clean = _masked_mean(out_clean.hidden_states[-1], clean_labels)
            h_noisy = _masked_mean(out_noisy.hidden_states[-1], noisy_labels)
            # Cosine in fp32 for numerical stability regardless of autocast.
            cons = (1.0 - F.cosine_similarity(h_clean.float(), h_noisy.float(), dim=-1)).mean()
            batch_loss = out_clean.loss + out_noisy.loss + beta * cons

        loss = batch_loss / grad_accum
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % grad_accum == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        total += batch_loss.item()
        bar.set_postfix({"ce_c": f"{out_clean.loss.item():.3f}",
                         "ce_n": f"{out_noisy.loss.item():.3f}",
                         "cons": f"{cons.item():.3f}"})
    return total / max(1, len(loader))


# --------------------------------------------------------------------------- #
# Generation / evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_answers(model, tokenizer, qa_df, ocr_df, device, noise_gen=None,
                     noise_type=None, level=2, batch_size=8, max_new_tokens=64,
                     num_beams=1, desc="Eval"):
    """Generate answers for a (noised) split; return (predictions, references)."""
    import pandas as pd
    from tqdm import tqdm

    model.eval()
    merged = pd.merge(qa_df, ocr_df[["image_id", "texts"]], on="image_id", how="inner")
    records = merged[["question", "texts", "answer"]].to_dict("records")

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # causal generation needs left padding
    preds, refs = [], []
    try:
        for i in tqdm(range(0, len(records), batch_size), desc=desc):
            chunk = records[i:i + batch_size]
            prompts = [
                _prompt_text(tokenizer, r["question"],
                             build_context(r["texts"], noise_gen, noise_type, level))
                for r in chunk
            ]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=1024, add_special_tokens=False).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                gen = model.generate(
                    **enc, max_new_tokens=max_new_tokens, num_beams=num_beams,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            preds.extend(d.strip() for d in decoded)
            refs.extend(str(r["answer"]).strip() for r in chunk)
    finally:
        tokenizer.padding_side = prev_side
    return preds, refs


def evaluate_anls(model, tokenizer, qa_df, ocr_df, device, **kwargs):
    """Convenience: generate then score with the shared ANLS metric."""
    preds, refs = generate_answers(model, tokenizer, qa_df, ocr_df, device, **kwargs)
    return compute_anls(preds, refs), preds, refs
