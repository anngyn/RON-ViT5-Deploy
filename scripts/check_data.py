"""Quick ReceiptVQA data sanity check before GPU training."""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import load_data


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "baseline.yaml"))
    parser.add_argument("--subset-ratio", type=float, default=0.001)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = resolve_project_path(config["data_dir"])
    (train_qa, train_ocr), (dev_qa, dev_ocr), (test_qa, test_ocr) = load_data(
        data_dir,
        args.subset_ratio,
    )

    print("Data check OK")
    print(f"  train: {len(train_qa)} QA, {len(train_ocr)} OCR images")
    print(f"  dev:   {len(dev_qa)} QA, {len(dev_ocr)} OCR images")
    print(f"  test:  {len(test_qa)} QA, {len(test_ocr)} OCR images")


if __name__ == "__main__":
    main()
