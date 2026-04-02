"""Training entrypoint for terrain segmentation."""

import argparse

from src.models.train import train_segmentation_model
from src.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a terrain segmentation model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    """Run training with config-defined parameters."""
    args = parse_args()
    config = load_yaml_config(args.config)
    training_cfg = config.get("training", {})
    output_model_path = config.get("paths", {}).get("model_output", "models/segmentation_model.txt")
    epochs = int(training_cfg.get("epochs", 5))

    model_path = train_segmentation_model(output_model_path=output_model_path, epochs=epochs)
    print(f"Model artifact saved to: {model_path}")


if __name__ == "__main__":
    main()
