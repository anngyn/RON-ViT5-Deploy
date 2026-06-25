"""Train ViT5 with noisy augmentation."""
import argparse
import os
import sys
import yaml
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import load_data
from src.dataset import TextOnlyVQADataset, NoisyVQADataset
from src.noise import OCRNoiseGenerator
from src.train import train_epoch_standard
from src.evaluate import evaluate, compute_anls


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def resolve_project_path(path):
    """Resolve config paths from project root."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def setup_logging(log_file):
    """Setup logging to file and console."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main(config_path):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for key in ['data_dir', 'output_dir', 'results_file', 'log_file']:
        config[key] = resolve_project_path(config[key])

    setup_logging(config['log_file'])
    logging.info(f"Config: {config_path}")
    logging.info(f"Method: {config['method']}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    # Load data
    (train_qa, train_ocr), (dev_qa, dev_ocr), (test_qa, test_ocr) = load_data(
        config['data_dir'], config['subset_ratio']
    )

    # Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config['model_name'], legacy=True)
    logging.info(f"Tokenizer: {tokenizer.name_or_path}")

    # Noise generator
    generator = OCRNoiseGenerator(seed=42)

    # Datasets - NOISY AUGMENTATION
    train_dataset = NoisyVQADataset(
        train_qa, train_ocr, tokenizer, generator,
        augmentation_ratio=config['augmentation_ratio'],
        noise_types=config['noise_types'],
        noise_level=config['noise_level'],
        noise_levels=config.get('noise_levels'),
        max_input_length=config['max_input_length'],
        max_output_length=config['max_output_length'],
        include_clean=True  # Clean + noisy
    )
    dev_dataset = TextOnlyVQADataset(
        dev_qa, dev_ocr, tokenizer,
        max_input_length=config['max_input_length'],
        max_output_length=config['max_output_length']
    )
    test_dataset = TextOnlyVQADataset(
        test_qa, test_ocr, tokenizer,
        max_input_length=config['max_input_length'],
        max_output_length=config['max_output_length']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    logging.info(f"Train samples: {len(train_dataset)} (clean + noisy)")
    logging.info(f"Train batches: {len(train_loader)}")

    # Model
    model = T5ForConditionalGeneration.from_pretrained(config['model_name'])
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable params: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Training loop
    best_dev_anls = -1.0
    os.makedirs(config['output_dir'], exist_ok=True)

    logging.info("=" * 60)
    logging.info("TRAINING: Noisy Augmentation")
    logging.info("=" * 60)

    for epoch in range(1, config['num_epochs'] + 1):
        logging.info(f"\nEpoch {epoch}/{config['num_epochs']}")
        logging.info("-" * 60)

        train_epoch_standard(model, train_loader, optimizer, tokenizer, device, epoch)

        preds_dev, refs_dev = evaluate(model, dev_loader, tokenizer, device, f"Dev Epoch {epoch}")
        dev_anls = compute_anls(preds_dev, refs_dev)
        logging.info(f"Dev ANLS: {dev_anls:.4f}")

        if dev_anls > best_dev_anls:
            best_dev_anls = dev_anls
            model.save_pretrained(config['output_dir'])
            tokenizer.save_pretrained(config['output_dir'])
            logging.info(f"✓ Best model saved (ANLS: {dev_anls:.4f})")

    logging.info(f"\nBest Dev ANLS: {best_dev_anls:.4f}")

    # Load best model
    model = T5ForConditionalGeneration.from_pretrained(config['output_dir'])
    model.to(device)

    # Evaluate on clean test
    logging.info("\n" + "=" * 60)
    logging.info("EVALUATION: Clean Test")
    logging.info("=" * 60)

    preds_test, refs_test = evaluate(model, test_loader, tokenizer, device, "Clean Test")
    test_anls_clean = compute_anls(preds_test, refs_test)
    logging.info(f"Clean Test ANLS: {test_anls_clean:.4f}")

    # Evaluate on noisy test sets
    noise_types = ['mixed', 'char', 'money', 'date', 'accent']
    results = [{'noise_type': 'clean', 'anls': test_anls_clean}]

    for noise_type in noise_types:
        logging.info(f"\nEvaluating {noise_type} noise...")
        noisy_dataset = NoisyVQADataset(
            test_qa, test_ocr, tokenizer, generator,
            augmentation_ratio=1.0, noise_types=[noise_type], noise_level=2,
            max_input_length=config['max_input_length'],
            max_output_length=config['max_output_length'],
            include_clean=False
        )
        noisy_loader = DataLoader(noisy_dataset, batch_size=16, shuffle=False)

        preds, refs = evaluate(model, noisy_loader, tokenizer, device, f"Noisy: {noise_type}")
        anls = compute_anls(preds, refs)
        logging.info(f"  ANLS: {anls:.4f}")
        results.append({'noise_type': noise_type, 'anls': anls})

    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(config['results_file']), exist_ok=True)
    results_df.to_csv(config['results_file'], index=False)
    logging.info(f"\n✓ Results saved to {config['results_file']}")

    # Print summary
    logging.info("\n" + "=" * 60)
    logging.info("RESULTS SUMMARY")
    logging.info("=" * 60)
    logging.info(f"{'Noise Type':<15} {'ANLS':>10}")
    logging.info("-" * 60)
    for r in results:
        logging.info(f"{r['noise_type']:<15} {r['anls']:>10.4f}")

    avg_noisy = sum(r['anls'] for r in results[1:]) / len(results[1:])
    logging.info("-" * 60)
    logging.info(f"{'Avg (noisy)':<15} {avg_noisy:>10.4f}")
    logging.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ViT5 with noisy augmentation.")
    parser.add_argument(
        '--config',
        default=os.path.join(os.path.dirname(__file__), '..', 'configs', 'noisy_aug.yaml'),
        help='Path to YAML config file.',
    )
    args = parser.parse_args()
    main(args.config)
