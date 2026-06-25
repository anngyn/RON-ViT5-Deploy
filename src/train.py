"""Training loops for different methods."""
import torch
from tqdm import tqdm
from src.models import consistency_loss

def _get_encoder(model):
    """Return encoder for wrapped or plain T5 models."""
    if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        return model.model.encoder
    if hasattr(model, 'encoder'):
        return model.encoder
    raise AttributeError("Model does not expose an encoder")


def train_epoch_standard(model, dataloader, optimizer, tokenizer, device, epoch_num):
    """
    Standard training loop (baseline, noisy aug, adapter-only).

    Args:
        model: model to train
        dataloader: DataLoader
        optimizer: optimizer
        tokenizer: tokenizer
        device: torch device
        epoch_num: current epoch number

    Returns:
        avg_loss: average loss for the epoch
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num}")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['src_attention_mask'].to(device)
        labels = batch['label_ids'].to(device)
        labels[labels == tokenizer.pad_token_id] = -100

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch_num} - Avg Loss: {avg_loss:.4f}")
    return avg_loss


def train_epoch_consistency(model, dataloader, optimizer, tokenizer, device, epoch_num, beta):
    """
    Training loop with consistency loss (consistency-only, RON-NACA).

    Args:
        model: model to train (must support return_encoder_hidden)
        dataloader: PairedVQADataset DataLoader
        optimizer: optimizer
        tokenizer: tokenizer
        device: torch device
        epoch_num: current epoch number
        beta: consistency loss weight

    Returns:
        avg_loss: average total loss
    """
    model.train()
    total_loss = 0
    total_ce_clean = 0
    total_ce_noisy = 0
    total_consistency = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num}")

    for batch in progress_bar:
        # Clean forward
        clean_input_ids = batch['clean_input_ids'].to(device)
        clean_attention_mask = batch['clean_attention_mask'].to(device)
        clean_labels = batch['clean_label_ids'].to(device)
        clean_labels[clean_labels == tokenizer.pad_token_id] = -100

        outputs_clean = model(
            input_ids=clean_input_ids,
            attention_mask=clean_attention_mask,
            labels=clean_labels
        )
        ce_loss_clean = outputs_clean.loss

        # Get encoder hidden states
        if hasattr(outputs_clean, 'encoder_hidden'):
            h_clean = outputs_clean.encoder_hidden
        else:
            # For base T5 without adapter (consistency-only method)
            encoder = _get_encoder(model)
            h_clean = encoder(
                input_ids=clean_input_ids,
                attention_mask=clean_attention_mask,
                return_dict=True
            ).last_hidden_state

        # Noisy forward
        noisy_input_ids = batch['noisy_input_ids'].to(device)
        noisy_attention_mask = batch['noisy_attention_mask'].to(device)
        noisy_labels = batch['noisy_label_ids'].to(device)
        noisy_labels[noisy_labels == tokenizer.pad_token_id] = -100

        outputs_noisy = model(
            input_ids=noisy_input_ids,
            attention_mask=noisy_attention_mask,
            labels=noisy_labels
        )
        ce_loss_noisy = outputs_noisy.loss

        if hasattr(outputs_noisy, 'encoder_hidden'):
            h_noisy = outputs_noisy.encoder_hidden
        else:
            encoder = _get_encoder(model)
            h_noisy = encoder(
                input_ids=noisy_input_ids,
                attention_mask=noisy_attention_mask,
                return_dict=True
            ).last_hidden_state

        # Consistency loss
        cons_loss = consistency_loss(h_clean, h_noisy, clean_attention_mask, noisy_attention_mask)

        # Total loss
        loss = ce_loss_noisy + ce_loss_clean + beta * cons_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce_clean += ce_loss_clean.item()
        total_ce_noisy += ce_loss_noisy.item()
        total_consistency += cons_loss.item()

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce_clean': f'{ce_loss_clean.item():.3f}',
            'ce_noisy': f'{ce_loss_noisy.item():.3f}',
            'cons': f'{cons_loss.item():.3f}'
        })

    n = len(dataloader)
    print(f"Epoch {epoch_num} Summary:")
    print(f"  Total Loss:  {total_loss/n:.4f}")
    print(f"  CE Clean:    {total_ce_clean/n:.4f}")
    print(f"  CE Noisy:    {total_ce_noisy/n:.4f}")
    print(f"  Consistency: {total_consistency/n:.4f}")

    return total_loss / n
