"""Evaluate Vintern-1B (Method A, text-only) on the clean + 14-noise grid.

Writes the same CSV schema as eval_noise_grid.py
(condition_id, noise_type, level, anls, drop_from_clean, model_tag) so the output is
directly usable by plot_noise_results.py and cross-model comparisons.

Loads the base Vintern model and, if --model-dir contains a LoRA adapter, applies it.
If no adapter is found it evaluates the base model zero-shot.
"""
import argparse
import os
import sys

import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import load_data
import vintern_common as vc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Same 14-condition grid and aliases as scripts/eval_noise_grid.py.
DEFAULT_NOISE_GRID = [
    ("N1", "accent_removal"), ("N2", "tone_confusion"), ("N3", "vowel_diacritic_confusion"),
    ("N4", "dd_confusion"), ("N5", "character_confusion"), ("N6", "glyph_confusion"),
    ("N7", "character_deletion"), ("N10", "token_deletion"), ("N13", "line_shuffle"),
    ("N14", "token_split"), ("N16", "money_noise"), ("N17", "date_noise"),
    ("N18", "code_noise"), ("N20", "mixed_noise"),
]
NAME_TO_ID = {name: cid for cid, name in DEFAULT_NOISE_GRID}
ID_TO_NAME = dict(DEFAULT_NOISE_GRID)


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def normalize_token(token):
    token = token.strip()
    if token in ID_TO_NAME:
        return token, ID_TO_NAME[token]
    if token in NAME_TO_ID:
        return NAME_TO_ID[token], token
    raise ValueError("Unknown noise token: %s" % token)


def maybe_load_adapter(base_llm, model_dir):
    """Attach a LoRA adapter if model_dir has one; otherwise return the base LM."""
    if model_dir and os.path.exists(os.path.join(model_dir, "adapter_config.json")):
        from peft import PeftModel
        print("Loading LoRA adapter from", model_dir)
        return PeftModel.from_pretrained(base_llm, model_dir)
    print("No adapter found -> zero-shot base model")
    return base_llm


def main():
    ap = argparse.ArgumentParser(description="Evaluate Vintern-1B on the OCR noise grid.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", help="LoRA adapter dir. Defaults to config output_dir.")
    ap.add_argument("--model-tag", help="Label written into the CSV.")
    ap.add_argument("--noise-types", nargs="*", default=[cid for cid, _ in DEFAULT_NOISE_GRID])
    ap.add_argument("--levels", nargs="*", type=int, default=[2])
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--max-eval-samples", type=int, default=None,
                    help="Evaluate on a fixed random subset of the test questions "
                         "(seed=42), so every condition and every flow use the same items "
                         "and stay comparable. Cuts the 15-condition grid from hours to "
                         "~1h on a T4. Default: full test set.")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = resolve(config["data_dir"])
    model_dir = resolve(args.model_dir or config["output_dir"])
    output_csv = resolve(args.output_csv)
    model_tag = args.model_tag or config.get("flow", "vintern")
    grid = [normalize_token(t) for t in args.noise_types]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print("Device:", device, "| model_tag:", model_tag)

    (_, _), (_, _), (test_qa, test_ocr) = load_data(data_dir, config["subset_ratio"])

    max_eval = args.max_eval_samples or config.get("eval_max_samples")
    if max_eval and len(test_qa) > max_eval:
        test_qa = test_qa.sample(n=max_eval, random_state=42).reset_index(drop=True)
        print("Eval subset: %d test questions (fixed seed=42)" % max_eval)
    else:
        print("Eval on full test set: %d questions" % len(test_qa))

    _, tokenizer, base_llm = vc.load_vintern(config["model_name"], device, dtype)
    model = maybe_load_adapter(base_llm, model_dir)

    def eval_condition(noise_type, level):
        anls, _, _ = vc.evaluate_anls(
            model, tokenizer, test_qa, test_ocr, device,
            noise_gen=(None if noise_type is None else vc.OCRNoiseGenerator(seed=42)),
            noise_type=noise_type, level=level,
            batch_size=args.batch_size, num_beams=args.num_beams,
            desc=(noise_type or "clean"),
        )
        return anls

    clean_anls = eval_condition(None, 0)
    print("Clean ANLS: %.4f" % clean_anls)
    rows = [{"condition_id": "clean", "noise_type": "clean", "level": 0,
             "anls": clean_anls, "drop_from_clean": 0.0, "model_tag": model_tag}]

    for cid, noise_type in grid:
        for level in args.levels:
            anls = eval_condition(noise_type, level)
            print("  %s / %s / L%d -> %.4f" % (cid, noise_type, level, anls))
            rows.append({"condition_id": cid, "noise_type": noise_type, "level": level,
                         "anls": anls, "drop_from_clean": clean_anls - anls, "model_tag": model_tag})

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print("Saved:", output_csv)


if __name__ == "__main__":
    main()
