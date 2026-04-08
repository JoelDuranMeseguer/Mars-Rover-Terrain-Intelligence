"""Inference entrypoint."""

import argparse


def main() -> None:
    """Parse args and print placeholder action."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(f"Running inference on {args.image} using {args.config}")


if __name__ == "__main__":
    main()
