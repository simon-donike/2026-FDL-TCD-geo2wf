"""Shared checkpoint loading and prediction adaptation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from geo2wf.config import instantiate_model
from geo2wf.models.base import PredictionBatch, PredictionRequest


class CheckpointLoader:
    """Strictly load old and new Lightning state dictionaries."""

    @staticmethod
    def load(
        config: dict[str, Any],
        checkpoint_path: str | Path,
        *,
        legacy_factory: Callable[[dict[str, Any]], Any] | None = None,
        strict: bool = True,
    ):
        model = instantiate_model(config, legacy_factory=legacy_factory)
        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        return model


class PredictionService:
    """Expose deterministic and ensemble models through one physical contract."""

    @staticmethod
    @torch.no_grad()
    def predict(
        model: Any,
        batch: dict[str, Any],
        request: PredictionRequest = PredictionRequest(),
    ) -> PredictionBatch:
        if callable(getattr(model, "predict_batch", None)):
            return model.predict_batch(batch, request)
        if callable(getattr(model, "sample_ensemble", None)):
            _, members = model.sample_ensemble(
                batch,
                num_samples=request.ensemble_size,
            )
            members = members.movedim(0, 1)
            offset = batch["target_norm_offset"].to(members)
            scale = batch["target_norm_scale"].to(members)
            shape = (members.shape[0], 1, members.shape[2]) + (1,) * (members.ndim - 3)
            members = offset.view(shape) + scale.view(shape) * members
            baseline = batch.get("_residual_diffusion_baseline_physical")
            return PredictionBatch(
                samples_physical=members,
                central_physical=members.mean(dim=1),
                baseline_physical=baseline,
            )
        if callable(getattr(model, "predict_physical", None)):
            prediction = model.predict_physical(batch)
            return PredictionBatch(
                samples_physical=prediction.unsqueeze(1),
                central_physical=prediction,
                baseline_physical=batch.get("era5_wind_speed_physical"),
            )
        raise TypeError(f"{type(model).__name__} does not implement prediction")
