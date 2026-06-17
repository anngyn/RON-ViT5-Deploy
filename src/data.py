"""Data loading helpers for ReceiptVQA."""
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


ANNOTATION_DIRS = (
    os.path.join("ReceiptVQA_annotations", "ReceiptVQA_annotations"),
    "ReceiptVQA_annotations",
)
OCR_DIRS = (
    os.path.join("features", "google_ocr", "google_ocr"),
    os.path.join("features", "google_ocr"),
)


def _resolve_data_dir(data_dir):
    def looks_like_dataset(path):
        return (
            os.path.isdir(path)
            and any(os.path.isdir(os.path.join(path, candidate)) for candidate in ANNOTATION_DIRS)
            and any(os.path.isdir(os.path.join(path, candidate)) for candidate in OCR_DIRS)
        )

    if looks_like_dataset(data_dir):
        return data_dir

    basename = os.path.basename(os.path.normpath(data_dir))
    candidates = []
    if basename:
        candidates.append(basename)
        parent = os.path.dirname(os.path.dirname(os.path.abspath(data_dir)))
        candidates.append(os.path.join(parent, basename))
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(data_dir)), "data", basename))
        candidates.append(os.path.join(parent, "data", basename))

    for fallback in candidates:
        if looks_like_dataset(fallback):
            logging.warning("Dataset not found at %s; using %s", data_dir, fallback)
            return fallback

    raise FileNotFoundError(f"Dataset not found or incomplete: {data_dir}")


def _find_existing_dir(data_dir, candidates, label):
    for candidate in candidates:
        path = os.path.join(data_dir, candidate)
        if os.path.isdir(path):
            return path
    checked = ", ".join(os.path.join(data_dir, candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing {label} directory. Checked: {checked}")


def _read_split(data_dir, split):
    annotation_dir = _find_existing_dir(data_dir, ANNOTATION_DIRS, "ReceiptVQA annotations")
    path = os.path.join(annotation_dir, f"ReceiptVQA_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing ReceiptVQA {split} annotations: {path}")
    return pd.read_csv(path)


def _load_ocr_for_images(data_dir, image_ids, split):
    ocr_dir = _find_existing_dir(data_dir, OCR_DIRS, "Google OCR")

    ocr_data = []
    missing = []

    for img_id in tqdm(image_ids, desc=f"Loading OCR ({split})"):
        ocr_path = os.path.join(ocr_dir, f"{img_id}.npy")
        if not os.path.exists(ocr_path):
            missing.append(img_id)
            continue

        ocr = np.load(ocr_path, allow_pickle=True).item()
        texts = ocr.get("texts", [])
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        ocr_data.append({"image_id": img_id, "texts": [str(text) for text in texts]})

    if missing:
        logging.warning("%s split missing OCR for %d images", split, len(missing))

    return pd.DataFrame(ocr_data, columns=["image_id", "texts"])


def _create_subset(qa_df, data_dir, split, ratio):
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"subset_ratio must be in (0, 1], got {ratio}")

    unique_imgs = qa_df["image_id"].dropna().unique()
    n_subset = max(1, int(len(unique_imgs) * ratio))
    sampled_imgs = np.random.choice(unique_imgs, size=n_subset, replace=False)

    qa_subset = qa_df[qa_df["image_id"].isin(sampled_imgs)].reset_index(drop=True)
    ocr_df = _load_ocr_for_images(data_dir, sampled_imgs, split)

    merged = pd.merge(qa_subset, ocr_df, on="image_id", how="inner")
    if merged.empty:
        raise ValueError(f"{split} split has no QA rows with matching OCR files")

    matched_qa = merged.drop(columns=["texts"])
    matched_ocr = ocr_df[ocr_df["image_id"].isin(merged["image_id"].unique())].reset_index(drop=True)

    logging.info(
        "%s: %d QA pairs, %d OCR images",
        split.capitalize(),
        len(matched_qa),
        len(matched_ocr),
    )
    return matched_qa.reset_index(drop=True), matched_ocr


def load_data(data_dir, subset_ratio=0.2):
    """Load and subsample ReceiptVQA train/dev/test splits."""
    data_dir = _resolve_data_dir(data_dir)

    np.random.seed(42)
    train_qa = _read_split(data_dir, "train")
    dev_qa = _read_split(data_dir, "dev")
    test_qa = _read_split(data_dir, "test")

    return (
        _create_subset(train_qa, data_dir, "train", subset_ratio),
        _create_subset(dev_qa, data_dir, "dev", subset_ratio),
        _create_subset(test_qa, data_dir, "test", subset_ratio),
    )
