"""Fine-tune Vintern-1B (Method A, text-only) under one of the three flows.

One script, parameterised by --flow, instead of three near-duplicate files:
    baseline    : LoRA SFT on clean OCR context
    noisy_aug   : LoRA SFT on randomly noised context
    consistency : LoRA SFT on clean/noisy pairs + hidden-state consistency

Reuses only src/data, src/noise, src/evaluate via scripts/vintern_common.py; it does
not modify any existing pipeline code. The heavy full-grid evaluation is done by
scripts/eval_noise_grid_vintern.py (called from the run_*.sh), so pass --skip-final-eval
from the flow runner.
"""
import argparse
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import load_data
from src.noise import OCRNoiseGenerator
import vintern_common as vc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FLOW_MODE = {"baseline": "clean", "noisy_aug": "noisy", "consistency": "paired"}


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for key in ["data_dir", "output_dir", "results_file", "log_file"]:
        config[key] = resolve(config[key])
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.num_epochs:
        config["num_epochs"] = args.num_epochs

    flow = args.flow or config.get("flow", "baseline")
    mode = FLOW_MODE[flow]

    setup_logging(config["log_file"])
    logging.info("Config: %s | Flow: %s | Model: %s", args.config, flow, config["model_name"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    logging.info("Device: %s", device)

    (train_qa, train_ocr), (dev_qa, dev_ocr), _ = load_data(config["data_dir"], config["subset_ratio"])

    wrapper, tokenizer, llm = vc.load_vintern(config["model_name"], device, dtype)
    logging.info("Loaded Vintern language model: %s", type(llm).__name__)

    if config.get("zero_shot", False):
        logging.info("zero_shot=true -> skipping training; eval uses the base model.")
        return

    model = vc.apply_lora(llm, config)

    noise_gen = OCRNoiseGenerator(seed=42)
    train_ds = vc.VinternQADataset(
        train_qa, train_ocr, tokenizer, mode=mode, noise_gen=noise_gen,
        noise_types=config.get("noise_types", ["mixed_noise"]),
        noise_levels=config.get("noise_levels", [config.get("noise_level", 2)]),
        max_length=config.get("max_input_length", 1024),
    )
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        collate_fn=vc.make_collate(tokenizer, mode),
    )
    logging.info("Train samples: %d (%s) | batches: %d", len(train_ds), mode, len(train_loader))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"]
    )
    grad_accum = config.get("grad_accum", 1)

    # Dev subset for checkpoint selection (generation is expensive).
    dev_n = min(config.get("dev_eval_samples", 300), len(dev_qa))
    dev_qa_small = dev_qa.head(dev_n).reset_index(drop=True)

    best_dev = -1.0
    os.makedirs(config["output_dir"], exist_ok=True)
    for epoch in range(1, config["num_epochs"] + 1):
        logging.info("Epoch %d/%d", epoch, config["num_epochs"])
        if mode == "paired":
            vc.train_epoch_causal_consistency(
                model, train_loader, optimizer, device, config.get("beta", 0.5), grad_accum, epoch
            )
        else:
            vc.train_epoch_causal(model, train_loader, optimizer, device, grad_accum, epoch)

        dev_anls, _, _ = vc.evaluate_anls(
            model, tokenizer, dev_qa_small, dev_ocr, device,
            batch_size=config.get("eval_batch_size", 8),
            num_beams=config.get("eval_num_beams", 1), desc=f"Dev {epoch}",
        )
        logging.info("Dev ANLS (n=%d): %.4f", dev_n, dev_anls)
        if dev_anls > best_dev:
            best_dev = dev_anls
            model.save_pretrained(config["output_dir"])       # LoRA adapter only
            tokenizer.save_pretrained(config["output_dir"])
            logging.info("Best adapter saved (ANLS %.4f) -> %s", dev_anls, config["output_dir"])

    logging.info("Best Dev ANLS: %.4f", best_dev)
    if args.skip_final_eval:
        logging.info("Skipping built-in eval; run_*.sh calls eval_noise_grid_vintern.py.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tune Vintern-1B (text-only, Method A).")
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "baseline_vintern.yaml"))
    ap.add_argument("--flow", choices=list(FLOW_MODE), help="Override config 'flow'.")
    ap.add_argument("--skip-final-eval", action="store_true")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--num-epochs", type=int)
    main(ap.parse_args())
