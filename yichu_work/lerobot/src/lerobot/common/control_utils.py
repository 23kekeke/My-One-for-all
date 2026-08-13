# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

########################################################################################
# Utilities
########################################################################################
import logging
import os
import select
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.policies import PreTrainedPolicy, prepare_observation_for_inference
from lerobot.utils.import_utils import _deepdiff_available, require_package

if TYPE_CHECKING or _deepdiff_available:
    from deepdiff import DeepDiff
else:
    DeepDiff = None

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots import Robot
from lerobot.types import PolicyAction

# Minimum interval (seconds) between consecutive intervention toggle presses.
INTERVENTION_TOGGLE_COOLDOWN_S = 0.5


def _stdin_supports_terminal_listener() -> bool:
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()


def _has_graphical_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _keyboard_debug_enabled() -> bool:
    return os.environ.get("LEROBOT_KEYBOARD_DEBUG", "").lower() in {"1", "true", "yes", "on"}


class TerminalKeyboardListener:
    """
    Keyboard listener that captures keys from the current terminal.

    Notes:
    - The terminal window must stay focused.
    - Only keys needed by the recorder loop are handled here.
    """

    def __init__(self, on_key):
        self.on_key = on_key
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._fd = None
        self._old_settings = None

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.2)

    def _read_escape_sequence(self) -> str | None:
        # Distinguish a bare Escape key from arrow-key escape sequences such as:
        # - CSI: ESC [ C
        # - SS3: ESC O C
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)
        if not ready:
            return "esc"

        seq = ""
        deadline = time.perf_counter() + 0.2
        while not self._stop_event.is_set() and time.perf_counter() < deadline:
            timeout_s = max(0.0, deadline - time.perf_counter())
            ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
            if not ready:
                break
            seq += os.read(self._fd, 32).decode(errors="ignore")
            if len(seq) >= 16:
                break

        key_map = {
            "[A": "up",
            "[B": "down",
            "[C": "right",
            "[D": "left",
            "OA": "up",
            "OB": "down",
            "OC": "right",
            "OD": "left",
        }
        if seq in key_map:
            return key_map[seq]

        if len(seq) >= 2 and seq[0] in ("[", "O") and seq[-1] in "ABCD":
            return key_map[f"[{seq[-1]}"]

        if _keyboard_debug_enabled():
            print(f"Unknown escape sequence: {seq!r}")

        # Ignore unrecognized escape sequences instead of treating them as a real Escape key.
        return None

    def _restore_terminal(self):
        if self._fd is not None and self._old_settings is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._fd = None
            self._old_settings = None

    def _run(self):
        if not _stdin_supports_terminal_listener():
            return

        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

        try:
            while not self._stop_event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue

                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    key_name = self._read_escape_sequence()
                    if key_name is not None:
                        self.on_key(key_name)
                elif ch:
                    # Pass through any other single character.
                    self.on_key(ch)
        finally:
            self._restore_terminal()


@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).

    Keyboard listening is handled separately through the current terminal, so this function only
    checks whether a graphical display is available.

    Returns:
        True if the environment is determined to be headless, False otherwise.
    """
    return not _has_graphical_display()


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    Performs a single-step inference to predict a robot action from an observation.

    This function encapsulates the full inference pipeline:
    1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
    2. Runs the preprocessor pipeline on the observation.
    3. Feeds the processed observation to the policy to get a raw action.
    4. Runs the postprocessor pipeline on the raw action.
    5. Formats the final action by removing the batch dimension and moving it to the CPU.

    Args:
        observation: A dictionary of NumPy arrays representing the robot's current observation.
        policy: The `PreTrainedPolicy` model to use for action prediction.
        device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
        preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
        postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
        use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
        task: An optional string identifier for the task.
        robot_type: An optional string identifier for the robot type.

    Returns:
        A `torch.Tensor` containing the predicted action, ready for the robot.
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        action = postprocessor(action)

    return action


def init_keyboard_listener():
    """
    Initializes a non-blocking keyboard listener for real-time user interaction.

    Key bindings:
    - 0: stop recording
    - 1: finish and save current episode
    - 2: discard current episode and rerecord
    - 3: start recording
    Arrow keys / Escape are kept as fallback aliases.

    This implementation listens directly from the current terminal, which is reliable
    on Ubuntu 22.04 / Wayland as long as the launching terminal stays focused.

    Returns:
        A tuple containing:
        - The listener instance, or `None` if keyboard listening is unavailable.
        - A dictionary of event flags (e.g., `exit_early`) that are set by key presses.
    """
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["start_recording"] = False

    def handle_key(key_name: str):
        # Normalize single-character keys to lowercase for reliable comparison.
        normalized = key_name.lower() if len(key_name) == 1 else key_name

        if normalized == "0" or key_name == "esc":
            print("Key '0' / Escape key pressed. Stopping data recording...")
            events["stop_recording"] = True
            events["exit_early"] = True
        elif normalized == "1" or key_name == "right":
            print("Key '1' / Right arrow key pressed. Finishing the current episode...")
            events["exit_early"] = True
        elif normalized == "2" or key_name == "left":
            print("Key '2' / Left arrow key pressed. Discarding the current episode and waiting to rerecord...")
            events["rerecord_episode"] = True
            events["exit_early"] = True
        elif normalized == "3":
            print("Key '3' pressed. Starting recording...")
            events["start_recording"] = True

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display will not be available."
        )
    if not _stdin_supports_terminal_listener():
        logging.warning(
            "stdin is not an interactive terminal. Keyboard shortcuts will be unavailable."
        )
        return None, events

    logging.info(
        "Using terminal keyboard listener. Controls: 0=stop, 1=finish/save, 2=rerecord, 3=start recording."
    )
    listener = TerminalKeyboardListener(handle_key)
    listener.start()

    return listener, events


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    Validates the dataset repository name against the presence of a policy configuration.

    This function enforces a naming convention: a dataset repository ID should start with "eval_"
    if and only if a policy configuration is provided for evaluation purposes.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        policy_cfg: The configuration object for the policy, or `None`.

    Raises:
        ValueError: If the naming convention is violated.
    """
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    Checks if a dataset's metadata is compatible with the current robot and recording setup.

    This function compares key metadata fields (`robot_type`, `fps`, and `features`) from the
    dataset against the current configuration to ensure that appended data will be consistent.

    Args:
        dataset: The `LeRobotDataset` instance to check.
        robot: The `Robot` instance representing the current hardware setup.
        fps: The current recording frequency (frames per second).
        features: The dictionary of features for the current recording session.

    Raises:
        ValueError: If any of the checked metadata fields do not match.
    """
    require_package("deepdiff", extra="deepdiff-dep")

    from lerobot.utils.constants import DEFAULT_FEATURES

    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
