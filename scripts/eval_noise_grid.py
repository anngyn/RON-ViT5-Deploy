"""Evaluate a trained model on clean + ReceiptVQA OCR noise grid."""
import argparse
import os
import sys
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import load_data
from src.dataset import TextOnlyVQADataset, NoisyVQADataset
from src.noise import OCRNoiseGenerator, NOISE_ALIASES
from src.evaluate import evaluate, compute_anls


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_NOISE_GRID = [
    ('N1', 'accent_removal'),
    ('N2', 'tone_confusion'),
    ('N3', 'vowel_diacritic_confusion'),
    ('N4', 'dd_confusion'),
    ('N5', 'character_confusion'),
    ('N6', 'glyph_confusion'),
    ('N7', 'character_deletion'),
    ('N10', 'token_deletion'),
    ('N13', 'line_shuffle'),
    ('N14', 'token_split'),
    ('N16', 'money_noise'),
    ('N17', 'date_noise'),
    ('N18', 'code_noise'),
    ('N20', 'mixed_noise'),
]
NOISE_ID_TO_NAME = dict(DEFAULT_NOISE_GRID)
NOISE_NAME_TO_ID = {name: cid for cid, name in DEFAULT_NOISE_GRID}


def resolve_project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def normalize_noise_token(token):
    token = token.strip()
    if token in NOISE_ID_TO_NAME:
        return token, NOISE_ID_TO_NAME[token]

    normalized = NOISE_ALIASES.get(token, token)
    if normalized in NOISE_NAME_TO_ID:
        return NOISE_NAME_TO_ID[normalized], normalized

    raise ValueError(f"Unknown noise token: {token}")


def parse_levels(values):
    levels = []
    for value in values:
        levels.append(int(value))
    return levels


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on ReceiptVQA OCR noise grid.")
    parser.add_argument('--config', required=True, help='Path to YAML config file.')
    parser.add_argument('--model-dir', help='Directory with saved model/tokenizer. Defaults to config output_dir.')
    parser.add_argument('--output-csv', required=True, help='Path to output CSV.')
    parser.add_argument('--model-tag', help='Label written into CSV.')
    parser.add_argument(
        '--noise-types',
        nargs='*',
        default=[cid for cid, _ in DEFAULT_NOISE_GRID],
        help='Noise ids or names. Defaults to N1 N2 N3 N4 N5 N6 N7 N10 N13 N14 N16 N17 N18 N20.',
    )
    parser.add_argument(
        '--levels',
        nargs='*',
        default=['2'],
        help='Noise levels to evaluate. Example: --levels 1 2 3',
    )
    parser.add_argument('--batch-size', type=int, default=16, help='Evaluation batch size.')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = resolve_project_path(config['data_dir'])
    model_dir = resolve_project_path(args.model_dir or config['output_dir'])
    output_csv = resolve_project_path(args.output_csv)
    model_tag = args.model_tag or config.get('method', os.path.basename(model_dir))
    levels = parse_levels(args.levels)

    selected_noise_grid = [normalize_noise_token(token) for token in args.noise_types]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model dir: {model_dir}")
    print(f"Model tag: {model_tag}")
    print(f"Noise grid: {[cid for cid, _ in selected_noise_grid]}")
    print(f"Levels: {levels}")

    (_, _), (_, _), (test_qa, test_ocr) = load_data(data_dir, config['subset_ratio'])

    tokenizer_source = model_dir if os.path.exists(os.path.join(model_dir, 'tokenizer_config.json')) else config['model_name']
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_source, legacy=True)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)

    clean_dataset = TextOnlyVQADataset(
        test_qa, test_ocr, tokenizer,
        max_input_length=config['max_input_length'],
        max_output_length=config['max_output_length']
    )
    clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False)

    preds_clean, refs_clean = evaluate(model, clean_loader, tokenizer, device, "Clean Test")
    clean_anls = compute_anls(preds_clean, refs_clean)
    print(f"Clean ANLS: {clean_anls:.4f}")

    generator = OCRNoiseGenerator(seed=42)
    rows = [{
        'condition_id': 'clean',
        'noise_type': 'clean',
        'level': 0,
        'anls': clean_anls,
        'drop_from_clean': 0.0,
        'model_tag': model_tag,
    }]

    for condition_id, noise_type in selected_noise_grid:
        for level in levels:
            print(f"Evaluating {condition_id} / {noise_type} / L{level}")
            noisy_dataset = NoisyVQADataset(
                test_qa, test_ocr, tokenizer, generator,
                augmentation_ratio=1.0,
                noise_types=[noise_type],
                noise_level=level,
                noise_levels=[level],
                max_input_length=config['max_input_length'],
                max_output_length=config['max_output_length'],
                include_clean=False
            )
            noisy_loader = DataLoader(noisy_dataset, batch_size=args.batch_size, shuffle=False)

            preds, refs = evaluate(model, noisy_loader, tokenizer, device, f"{condition_id}-L{level}")
            anls = compute_anls(preds, refs)
            rows.append({
                'condition_id': condition_id,
                'noise_type': noise_type,
                'level': level,
                'anls': anls,
                'drop_from_clean': clean_anls - anls,
                'model_tag': model_tag,
            })

    result_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    print(f"Saved CSV: {output_csv}")
    noisy_rows = result_df[result_df['condition_id'] != 'clean'].sort_values('drop_from_clean', ascending=False)
    print(noisy_rows[['condition_id', 'noise_type', 'level', 'anls', 'drop_from_clean']].to_string(index=False))


if __name__ == '__main__':
    main()
