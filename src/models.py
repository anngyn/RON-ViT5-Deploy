"""Model architectures: Adapter, RON-NACA."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration


class BottleneckAdapter(nn.Module):
    """Lightweight bottleneck adapter."""

    def __init__(self, d_model=768, bottleneck_dim=64):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, d_model)

        # Initialize near-zero for residual stability
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return residual + x


class T5WithAdapter(nn.Module):
    """T5 with adapter at encoder last layer."""

    def __init__(self, base_model_name="VietAI/vit5-base", freeze_base=True, bottleneck_dim=64):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(base_model_name)

        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

        d_model = self.model.config.d_model
        self.adapter = BottleneckAdapter(d_model=d_model, bottleneck_dim=bottleneck_dim)

    def forward(self, input_ids, attention_mask, labels=None, return_encoder_hidden=False):
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        hidden_states = encoder_outputs.last_hidden_state
        hidden_states = self.adapter(hidden_states)

        if return_encoder_hidden:
            return hidden_states

        if labels is not None:
            decoder_outputs = self.model.decoder(
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            lm_logits = self.model.lm_head(decoder_outputs.last_hidden_state)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            return type('Outputs', (), {'loss': loss, 'encoder_hidden': hidden_states})()
        else:
            return type('Outputs', (), {'last_hidden_state': hidden_states})()

    def generate(self, input_ids, attention_mask, **kwargs):
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = self.adapter(encoder_outputs.last_hidden_state)

        return self.model.generate(
            encoder_outputs=type('Outputs', (), {'last_hidden_state': hidden_states})(),
            attention_mask=attention_mask,
            **kwargs
        )


class RON_NACA(nn.Module):
    """RON-NACA: Adapter + Consistency Loss."""

    def __init__(self, base_model_name="VietAI/vit5-base", freeze_base=True, bottleneck_dim=64):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(base_model_name)

        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

        d_model = self.model.config.d_model
        self.adapter = BottleneckAdapter(d_model=d_model, bottleneck_dim=bottleneck_dim)

    def forward(self, input_ids, attention_mask, labels=None, return_encoder_hidden=False):
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        hidden_states = encoder_outputs.last_hidden_state
        hidden_states = self.adapter(hidden_states)

        if return_encoder_hidden:
            return hidden_states

        if labels is not None:
            decoder_outputs = self.model.decoder(
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            lm_logits = self.model.lm_head(decoder_outputs.last_hidden_state)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            return type('Outputs', (), {'loss': loss, 'encoder_hidden': hidden_states})()
        else:
            return type('Outputs', (), {'last_hidden_state': hidden_states})()

    def generate(self, input_ids, attention_mask, **kwargs):
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = self.adapter(encoder_outputs.last_hidden_state)

        return self.model.generate(
            encoder_outputs=type('Outputs', (), {'last_hidden_state': hidden_states})(),
            attention_mask=attention_mask,
            **kwargs
        )


def consistency_loss(h_clean, h_noisy, mask_clean, mask_noisy):
    """
    Cosine similarity loss between clean and noisy encoder representations.

    Args:
        h_clean: [batch, seq_len, d_model]
        h_noisy: [batch, seq_len, d_model]
        mask_clean: [batch, seq_len]
        mask_noisy: [batch, seq_len]

    Returns:
        loss: scalar (1 - cosine_similarity)
    """
    # Mean pool over sequence (masked)
    mask_clean = mask_clean.unsqueeze(-1).float()
    mask_noisy = mask_noisy.unsqueeze(-1).float()

    h_clean_pooled = (h_clean * mask_clean).sum(dim=1) / mask_clean.sum(dim=1).clamp(min=1e-9)
    h_noisy_pooled = (h_noisy * mask_noisy).sum(dim=1) / mask_noisy.sum(dim=1).clamp(min=1e-9)

    # Cosine similarity
    cos_sim = F.cosine_similarity(h_clean_pooled, h_noisy_pooled, dim=-1)

    # Loss: 1 - similarity (minimize = maximize similarity)
    loss = (1.0 - cos_sim).mean()
    return loss
