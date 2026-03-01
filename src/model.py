"""Dual-head safety classifier: binary (safe/unsafe) + multi-label categories."""

import torch
import torch.nn as nn
from transformers import AutoModel


class SafetyClassifier(nn.Module):
    def __init__(self, base_model_name: str, num_categories: int = 7):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden = self.backbone.config.hidden_size  # 384 for xsmall

        self.binary_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )
        self.category_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, num_categories),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.last_hidden_state[:, 0]  # [CLS] token

        binary_logit = self.binary_head(cls_embed)      # (B, 1)
        category_logits = self.category_head(cls_embed)  # (B, 7)

        return binary_logit, category_logits

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        self.eval()
        with torch.no_grad():
            binary_logit, category_logits = self.forward(input_ids, attention_mask)

        unsafe_score = torch.sigmoid(binary_logit).squeeze(-1)  # (B,)
        category_scores = torch.sigmoid(category_logits)         # (B, 7)

        return {
            "unsafe_score": unsafe_score,
            "category_scores": category_scores,
        }
