#!/usr/bin/env python3
"""Visualize left arm joint positions logged by x_robot.py."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot left arm joint positions")
    parser.add_argument(
        "log_file",
        nargs="?",
        # default="/tmp/left_arm_pos_log.txt",
        default="/tmp/left_arm_pos_clipped_log.txt",
        help="Path to log file (default: /tmp/left_arm_pos_log.txt)",
    )
    parser.add_argument("--save", "-s", type=str, help="Save plot to file instead of showing")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.log_file, sep="\t")
    except FileNotFoundError:
        print(f"Log file not found: {args.log_file}")
        print("Run the robot first to generate the log file.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in df.columns:
        if col == "step":
            continue
        ax.plot(df["step"], df[col], label=col)

    ax.set_xlabel("Step")
    ax.set_ylabel("Joint position")
    ax.set_title("Left Arm Joint Positions")
    ax.legend(loc="best")
    ax.grid(True)

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
