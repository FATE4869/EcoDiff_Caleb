import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def extract_ntk_losses(log_path):
    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    text = log_path.read_text(errors="ignore")

    # Match lines like:
    # ntk_loss: 0.006113
    pattern = re.compile(r"ntk_loss:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

    ntk_losses = [float(x) for x in pattern.findall(text)]

    return ntk_losses


def plot_ntk_losses(ntk_losses, save_path="ntk_loss_plot.png"):
    if len(ntk_losses) == 0:
        raise ValueError("No ntk_loss values found in the log file.")

    steps = list(range(len(ntk_losses)))

    plt.figure(figsize=(8, 5))
    plt.plot(steps, ntk_losses, marker="o")
    plt.xlabel("NTK loss index")
    plt.ylabel("ntk_loss")
    plt.title("Extracted NTK Loss")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to the training log file",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="ntk_loss_plot.png",
        help="Path to save the plot",
    )
    args = parser.parse_args()

    ntk_losses = extract_ntk_losses(args.log)

    print(f"Found {len(ntk_losses)} ntk_loss values")
    print(ntk_losses)

    plot_ntk_losses(ntk_losses, args.save)


if __name__ == "__main__":
    main()