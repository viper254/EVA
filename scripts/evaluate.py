"""Evaluate an EVA's developmental progress.

Usage:
    python scripts/evaluate.py --checkpoint path/to/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import torch

from eva.core.baby_brain import BabyBrain, detect_device
from eva.core.config import EVAConfig
from eva.core.tokenizer import EVATokenizer
from eva.environment.nursery import NurseryEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_prediction(
    brain: BabyBrain,
    tokenizer: EVATokenizer,
    test_strings: list[str],
) -> dict[str, float]:
    """Evaluate prediction accuracy on test strings."""
    brain.eval()
    total_correct = 0
    total_tokens = 0

    for test_str in test_strings:
        tokens = tokenizer.encode(test_str)
        if len(tokens) < 3:
            continue

        for i in range(2, len(tokens)):
            input_ids = torch.tensor([tokens[:i]], dtype=torch.long)
            with torch.no_grad():
                predicted_dist = brain.predict_next(input_ids)
            predicted_token = predicted_dist.argmax(dim=-1).item()
            if predicted_token == tokens[i]:
                total_correct += 1
            total_tokens += 1

    accuracy = total_correct / max(1, total_tokens)
    return {
        "accuracy": accuracy,
        "total_correct": total_correct,
        "total_tokens": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EVA development")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    config = EVAConfig.from_yaml(args.config)
    tokenizer = EVATokenizer()

    # Detect device
    device_str = getattr(config.hardware, "device", "auto")
    if device_str == "auto":
        device = detect_device()
    else:
        device = torch.device(device_str)
    logger.info("Using device: %s", device)

    # Load brain
    brain = BabyBrain(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        dtype_str=config.model.dtype,
        device=device,
    )
    checkpoint = torch.load(
        args.checkpoint, weights_only=False, map_location=device
    )
    brain.load_state_dict(checkpoint["brain_state_dict"])

    # Test strings of increasing difficulty
    test_strings = [
        "aaaaaaaaaa",
        "ababababab",
        "abcabcabc",
        "hello hello hello",
        "the cat sat on the mat",
    ]

    logger.info("Evaluating prediction accuracy...")
    results = evaluate_prediction(brain, tokenizer, test_strings)

    # Print checkpoint info
    print("\n=== EVA Evaluation Report ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Step: {checkpoint.get('step', 'unknown')}")
    print(f"Phase: {checkpoint.get('curriculum', {}).get('current_phase', 'unknown')}")
    print(f"\nPrediction accuracy: {results['accuracy']:.4f}")
    print(f"Correct: {results['total_correct']}/{results['total_tokens']}")

    if "affect" in checkpoint:
        print(f"\nAffective state: {checkpoint['affect']}")
    if "homeostasis" in checkpoint:
        print(f"Homeostasis: {checkpoint['homeostasis']}")

    print("\n=== End Report ===")


if __name__ == "__main__":
    main()
