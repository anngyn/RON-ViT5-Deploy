"""Evaluation utilities and ANLS metric."""
import torch
from tqdm import tqdm


def levenshtein_distance(s1, s2):
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def anls_score(prediction, ground_truth, threshold=0.5):
    """Compute ANLS (Average Normalized Levenshtein Similarity)."""
    pred = prediction.lower().strip()
    gt = ground_truth.lower().strip()

    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0

    dist = levenshtein_distance(pred, gt)
    similarity = 1.0 - (dist / max(len(pred), len(gt)))

    return similarity if similarity >= threshold else 0.0


def compute_anls(predictions, ground_truths, threshold=0.5):
    """Compute average ANLS for a list of predictions."""
    scores = [anls_score(pred, gt, threshold) for pred, gt in zip(predictions, ground_truths)]
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(model, dataloader, tokenizer, device, desc="Evaluation"):
    """
    Evaluate model on a dataset.

    Args:
        model: model to evaluate
        dataloader: DataLoader
        tokenizer: tokenizer
        device: torch device
        desc: description for progress bar

    Returns:
        predictions: list of predicted strings
        ground_truths: list of ground truth strings
    """
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['src_attention_mask'].to(device)

            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )

            # Decode predictions and labels
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            labels = batch['label_ids'].numpy()
            labels[labels == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(preds)
            ground_truths.extend(refs)

    return predictions, ground_truths
