"""Train baseline ViT5 on clean data."""
import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import TextOnlyVQADataset, NoisyVQADataset
from src.noise import OCRNoiseGenerator
from src.train import train_epoch_standard
from src.evaluate import evaluate, compute_anls


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


def load_data(data_dir, subset_ratio=0.2):
    """Load and subsample ReceiptVQA dataset."""
    train_qa = pd.read_csv(f'{data_dir}/ReceiptVQA_annotations/ReceiptVQA_annotations/ReceiptVQA_train.csv')
    dev_qa = pd.read_csv(f'{data_dir}/ReceiptVQA_annotations/ReceiptVQA_annotations/ReceiptVQA_dev.csv')
    test_qa = pd.read_csv(f'{data_dir}/ReceiptVQA_annotations/ReceiptVQA_annotations/ReceiptVQA_test.csv')

    ocr_dir = f'{data_dir}/features/google_ocr/google_ocr'

    def create_subset(qa_df, ratio):
        unique_imgs = qa_df['image_id'].unique()
        n_subset = int(len(unique_imgs) * ratio)
        sampled_imgs = np.random.choice(unique_imgs, size=n_subset, replace=False)

        qa_subset = qa_df[qa_df['image_id'].isin(sampled_imgs)].reset_index(drop=True)

        ocr_data = []
        for img_id in tqdm(sampled_imgs, desc="Loading OCR"):
            ocr_path = f"{ocr_dir}/{img_id}.npy"
            if os.path.exists(ocr_path):
                ocr = np.load(ocr_path, allow_pickle=True).item()
                ocr_data.append({'image_id': img_id, 'texts': ocr['texts']})

        ocr_df = pd.DataFrame(ocr_data)
        return qa_subset, ocr_df

    np.random.seed(42)
    train_qa, train_ocr = create_subset(train_qa, subset_ratio)
    dev_qa, dev_ocr = create_subset(dev_qa, subset_ratio)
    test_qa, test_ocr = create_subset(test_qa, subset_ratio)

    logging.info(f"Train: {len(train_qa)} QA pairs")
    logging.info(f"Dev:   {len(dev_qa)} QA pairs")
    logging.info(f"Test:  {len(test_qa)} QA pairs")

    return (train_qa, train_ocr), (dev_qa, dev_ocr), (test_qa, test_ocr)


def main(config_path):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

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

    # Datasets
    train_dataset = TextOnlyVQADataset(
        train_qa, train_ocr, tokenizer,
        max_input_length=config['max_input_length'],
        max_output_length=config['max_output_length']
    )
    dev_dataset = TextOnlyVQADataset(dev_qa, dev_ocr, tokenizer)
    test_dataset = TextOnlyVQADataset(test_qa, test_ocr, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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
    logging.info("TRAINING: Baseline")
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

    generator = OCRNoiseGenerator(seed=42)

    for noise_type in noise_types:
        logging.info(f"\nEvaluating {noise_type} noise...")
        noisy_dataset = NoisyVQADataset(
            test_qa, test_ocr, tokenizer, generator,
            augmentation_ratio=0.0, noise_types=[noise_type], noise_level=2,
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
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'baseline.yaml')
    main(config_path)
