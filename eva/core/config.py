"""EVA Configuration — load, validate, and access all parameters."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigSection:
    """Dot-accessible configuration section."""

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigSection(value))
            elif isinstance(value, list):
                setattr(self, key, [
                    ConfigSection(item) if isinstance(item, dict) else item
                    for item in value
                ])
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return f"ConfigSection({self.__dict__})"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigSection):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, ConfigSection) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


class EVAConfig:
    """Complete EVA configuration with validation and memory estimation.

    Loads from YAML, validates Ron Protocol invariants, and provides
    typed dot-access to all parameters.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._raw = data
        self.model = ConfigSection(data.get("model", {}))
        self.hardware = ConfigSection(data.get("hardware", {}))
        self.curiosity = ConfigSection(data.get("curiosity", {}))
        self.emotions = ConfigSection(data.get("emotions", {}))
        self.guidance = ConfigSection(data.get("guidance", {}))
        self.legacy = ConfigSection(data.get("legacy", {}))
        self.identity = ConfigSection(data.get("identity", {}))
        self.reproduction = ConfigSection(data.get("reproduction", {}))
        self.portage = ConfigSection(data.get("portage", {}))
        self.training = ConfigSection(data.get("training", {}))
        self.developmental_emotions = ConfigSection(
            data.get("developmental_emotions", {})
        )
        self.memory = ConfigSection(data.get("memory", {}))
        self.novelty = ConfigSection(data.get("novelty", {}))

    @classmethod
    def from_yaml(cls, path: str) -> EVAConfig:
        """Load configuration from a YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        config = cls(data)
        config.validate()
        return config

    def validate(self) -> None:
        """Validate all configuration parameters, enforcing Ron Protocol."""
        errors: list[str] = []

        # Model validation
        if not hasattr(self.model, "d_model") or self.model.d_model <= 0:
            errors.append("model.d_model must be a positive integer")
        if not hasattr(self.model, "n_layers") or self.model.n_layers <= 0:
            errors.append("model.n_layers must be a positive integer")
        if not hasattr(self.model, "n_heads") or self.model.n_heads <= 0:
            errors.append("model.n_heads must be a positive integer")
        if hasattr(self.model, "d_model") and hasattr(self.model, "n_heads"):
            if self.model.d_model % self.model.n_heads != 0:
                errors.append(
                    f"model.d_model ({self.model.d_model}) must be divisible "
                    f"by model.n_heads ({self.model.n_heads})"
                )
        if not hasattr(self.model, "random_init") or not self.model.random_init:
            errors.append(
                "RON PROTOCOL VIOLATION: model.random_init must be true. "
                "No pretrained weights allowed."
            )

        # Curiosity weights should sum to ~1.0
        if all(
            hasattr(self.curiosity, attr)
            for attr in ["alpha", "beta", "gamma", "delta"]
        ):
            total = (
                self.curiosity.alpha
                + self.curiosity.beta
                + self.curiosity.gamma
                + self.curiosity.delta
            )
            if abs(total - 1.0) > 0.01:
                logger.warning(
                    "Curiosity weights sum to %.3f (expected ~1.0)", total
                )

        # Ron Protocol: contradiction must be null
        if hasattr(self.legacy, "contradiction"):
            if hasattr(self.legacy.contradiction, "prioritize"):
                if self.legacy.contradiction.prioritize is not None:
                    errors.append(
                        "RON PROTOCOL VIOLATION: "
                        "legacy.contradiction.prioritize must be null. "
                        "EVA must decide for itself."
                    )

        # Ron Protocol: children get fresh weights
        if hasattr(self.reproduction, "inheritance"):
            if hasattr(self.reproduction.inheritance, "weights"):
                if self.reproduction.inheritance.weights:
                    errors.append(
                        "RON PROTOCOL VIOLATION: "
                        "reproduction.inheritance.weights must be false. "
                        "Children get fresh random weights."
                    )

        # Memory estimation warning
        mem_gb = self.estimate_memory_gb()
        max_ram = getattr(self.hardware, "max_ram_gb", 4)
        if mem_gb > max_ram:
            logger.warning(
                "Estimated memory %.2f GB exceeds hardware.max_ram_gb (%d GB)",
                mem_gb,
                max_ram,
            )

        if errors:
            raise ValueError(
                "Configuration validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.info("Configuration validated successfully.")

    def estimate_memory_gb(self) -> float:
        """Estimate model memory usage in GB.

        Rough estimate: param_count * bytes_per_param.
        Transformer params ≈ 12 * n_layers * d_model^2 (approximate).
        """
        d_model = getattr(self.model, "d_model", 768)
        n_layers = getattr(self.model, "n_layers", 12)
        vocab_size = getattr(self.model, "vocab_size", 512)
        dtype = getattr(self.model, "dtype", "float16")

        # Approximate parameter count
        # Embedding: vocab_size * d_model
        # Per layer: 4 * d_model^2 (attention) + 8 * d_model^2 (ffn) = 12 * d_model^2
        # Output projection: d_model * vocab_size
        embedding_params = vocab_size * d_model
        layer_params = 12 * d_model * d_model * n_layers
        output_params = d_model * vocab_size
        total_params = embedding_params + layer_params + output_params

        bytes_per_param = 2 if dtype == "float16" else 4
        memory_bytes = total_params * bytes_per_param
        memory_gb = memory_bytes / (1024 ** 3)

        return memory_gb

    def to_dict(self) -> dict[str, Any]:
        """Return full configuration as a dictionary."""
        return self._raw.copy()
