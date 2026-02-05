"""Transformer-based forecaster scaffolding."""
from __future__ import annotations

from typing import Sequence


class TransformerForecaster:
    """Bare-bones transformer forecaster.

    This class expects vectorized sequences. Wire it to your tokenizer or
    feature windows and implement the training loop.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or "cpu"
        self._model = None
        self._head = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers/torch not installed. Install with `pip install -e .[ai]`."
            ) from exc

        config = AutoConfig.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        head = torch.nn.Linear(config.hidden_size, 1)
        model.to(self.device)
        head.to(self.device)
        self._model = model
        self._head = head

    def fit(self, *_: object, **__: object) -> None:
        raise NotImplementedError(
            "Training loop not implemented. Use PyTorch Lightning or a custom loop."
        )

    def predict(self, embeddings: Sequence[Sequence[float]]) -> list[float]:
        self._ensure_model()
        import torch

        model = self._model
        head = self._head
        if model is None or head is None:
            raise RuntimeError("Model not initialized")

        with torch.no_grad():
            inputs = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            outputs = model(inputs_embeds=inputs.unsqueeze(1))
            pooled = outputs.last_hidden_state[:, 0, :]
            scores = head(pooled).squeeze(-1)
            return scores.detach().cpu().numpy().tolist()
