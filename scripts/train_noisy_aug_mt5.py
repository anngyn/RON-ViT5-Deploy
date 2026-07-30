"""Train with noisy augmentation, any seq2seq backbone (BARTpho/mT5/...).

Copy of train_noisy_aug.py with T5-specific tokenizer/model classes swapped
for the generic Auto* classes. Does not touch the original ViT5 script.
"""
import argparse
import os
import random
import sys
import yaml
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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


def main(config_path, skip_final_eval=False, batch_size_override=None, learning_rate_override=None, num_epochs_override=None):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_seed = config.get('seed', 42)
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    if batch_size_override:
        config['batch_size'] = batch_size_override
    if learning_rate_override:
        config['learning_rate'] = learning_rate_override
    if num_epochs_override:
        config['num_epochs'] = num_epochs_override

    for key in ['data_dir', 'output_dir', 'results_file', 'log_file']:
        config[key] = resolve_project_path(config[key])

    setup_logging(config['log_file'])
    logging.info(f"Config: {config_path}")
    logging.info(f"Method: {config['method']}")
    logging.info(f"Model: {config['model_name']}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    # Load data
    (train_qa, train_ocr), (dev_qa, dev_ocr), (test_qa, test_ocr) = load_data(
        config['data_dir'], config['subset_ratio']
    )

    # Tokenizer (Auto resolves to BartphoTokenizer / T5Tokenizer / etc.)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], legacy=True)
    logging.info(f"Tokenizer: {tokenizer.name_or_path} ({type(tokenizer).__name__})")

    # Datasets - NOISY AUGMENTATION
    def build_train_dataset(epoch_index=0):
        epoch_seed = base_seed + epoch_index
        return NoisyVQADataset(
            train_qa, train_ocr, tokenizer,
            OCRNoiseGenerator(seed=epoch_seed),
            augmentation_ratio=config['augmentation_ratio'],
            noise_types=config['noise_types'],
            noise_level=config['noise_level'],
            noise_levels=config.get('noise_levels'),
            max_input_length=config['max_input_length'],
            max_output_length=config['max_output_length'],
            include_clean=config.get('include_clean', True),
            total_size_ratio=config.get('total_size_ratio'),
            clean_ratio=config.get('clean_ratio'),
            sample_seed=epoch_seed,
            choice_seed=epoch_seed,
        )

    train_dataset = build_train_dataset()
    dev_dataset = TextOnlyVQADataset(
        dev_qa, dev_ocr, tokenizer,
        max_input_length=config['max_input_length'],
        max_output_length=config['max_output_length']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

    logging.info(f"Train samples: {len(train_dataset)} (clean + noisy)")
    logging.info(f"Train batches: {len(train_loader)}")
    if config.get('total_size_ratio') is not None:
        logging.info(
            "Fixed-budget mix: total=%.1fx, clean=%.0f%%, noisy=%.0f%%",
            config['total_size_ratio'],
            100 * config['clean_ratio'],
            100 * (1 - config['clean_ratio']),
        )

    # Model (Auto resolves to MBartForConditionalGeneration / T5ForConditionalGeneration / etc.)
    model = AutoModelForSeq2SeqLM.from_pretrained(config['model_name'])
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable params: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Training loop
    best_dev_anls = -1.0
    os.makedirs(config['output_dir'], exist_ok=True)

    logging.info("=" * 60)
    logging.info(f"TRAINING: Noisy Augmentation ({config['model_name']})")
    logging.info("=" * 60)

    for epoch in range(1, config['num_epochs'] + 1):
        if epoch > 1 and config.get('dynamic_augmentation', False):
            train_dataset = build_train_dataset(epoch - 1)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
            )
            logging.info(
                "Rebuilt epoch %d mix with seed %d (%d samples)",
                epoch,
                base_seed + epoch - 1,
                len(train_dataset),
            )

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

    if skip_final_eval:
        logging.info("Skipping final test/noise evaluation; flow runner handles it.")
        return

    # Load best model
    model = AutoModelForSeq2SeqLM.from_pretrained(config['output_dir'])
    model.to(device)

    test_dataset = TextOnlyVQADataset(
        test_qa, test_ocr, tokenizer,
        max_input_length=config['max_input_length'],
        max_output_length=config['max_output_length']
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Evaluate on clean test
    logging.info("\n" + "=" * 60)
    logging.info("EVALUATION: Clean Test")
    logging.info("=" * 60)

    preds_test, refs_test = evaluate(model, test_loader, tokenizer, device, "Clean Test")
    test_anls_clean = compute_anls(preds_test, refs_test)
    logging.info(f"Clean Test ANLS: {test_anls_clean:.4f}")

    # Evaluate on noisy test sets
    generator = OCRNoiseGenerator(seed=base_seed)
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
    parser = argparse.ArgumentParser(description="Train with noisy augmentation (generic seq2seq backbone).")
    parser.add_argument(
        '--config',
        default=os.path.join(os.path.dirname(__file__), '..', 'configs', 'noisy_aug_mt5.yaml'),
        help='Path to YAML config file.',
    )
    parser.add_argument(
        '--skip-final-eval',
        action='store_true',
        help='Skip built-in clean/noisy test evaluation after training.',
    )
    parser.add_argument('--batch-size', type=int, help='Override config batch_size')
    parser.add_argument('--learning-rate', type=float, help='Override config learning_rate')
    parser.add_argument('--num-epochs', type=int, help='Override config num_epochs')
    args = parser.parse_args()
    main(args.config, skip_final_eval=args.skip_final_eval,
         batch_size_override=args.batch_size,
         learning_rate_override=args.learning_rate,
         num_epochs_override=args.num_epochs)
