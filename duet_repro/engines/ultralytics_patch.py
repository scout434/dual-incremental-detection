"""Stable package import for the DuET-aware Ultralytics trainer patch."""

from __future__ import annotations

from duet_repro.engines.duet_trainer import DuETDetectionTrainer

__all__ = ["DuETDetectionTrainer"]
