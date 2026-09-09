"""VisualWebArena imports and version-specific action/evaluator bindings."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VWABindings:
    """VWA APIs used by the vision-only benchmark runner."""

    action_types: Any
    script_browser_env: Any
    create_none_action: Any
    create_keyboard_type_action: Any
    create_key_press_action: Any
    create_scroll_action: Any
    create_stop_action: Any
    create_go_back_action: Any
    create_go_forward_action: Any
    renew_comb: Any
    evaluator_router: Any
    image_utils: Any


class BrowserEnvActionFactory:
    """Create low-level coordinate actions understood by VWA's browser_env."""

    def __init__(
        self, bindings: VWABindings, viewport_width: int, viewport_height: int
    ) -> None:
        if viewport_width < 1 or viewport_height < 1:
            raise ValueError("Viewport dimensions must be positive")
        self.bindings = bindings
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

    def click(self, x: int, y: int) -> Any:
        # VWA's create_mouse_click_action treats zero as false, so construct the
        # action explicitly to preserve valid top/left-edge coordinates.
        action = self.bindings.create_none_action()
        action.update(
            {
                "action_type": self.bindings.action_types.MOUSE_CLICK,
                # VWA low-level MOUSE_CLICK stores normalized coordinates and
                # multiplies them by page.viewport_size during execution.
                "coords": np.array(
                    [x / self.viewport_width, y / self.viewport_height],
                    dtype=np.float32,
                ),
            }
        )
        return action

    def hover(self, x: int, y: int) -> Any:
        action = self.bindings.create_none_action()
        action.update(
            {
                "action_type": self.bindings.action_types.MOUSE_HOVER,
                "coords": np.array(
                    [x / self.viewport_width, y / self.viewport_height],
                    # VWA's beartyped hover executor rejects numpy.float32.
                    dtype=np.float64,
                ),
            }
        )
        return action

    def type_text(self, text: str) -> Any:
        return self.bindings.create_keyboard_type_action(text)

    def press(self, key: str) -> Any:
        return self.bindings.create_key_press_action(key)

    def scroll(self, direction: str) -> Any:
        return self.bindings.create_scroll_action(direction)

    def go_back(self) -> Any:
        return self.bindings.create_go_back_action()

    def go_forward(self) -> Any:
        return self.bindings.create_go_forward_action()

    def stop(self, answer: str) -> Any:
        return self.bindings.create_stop_action(answer)


class EvaluationCaptioner:
    """Load BLIP-2 only for VWA's page-image evaluator, never for agent input."""

    def __init__(self, bindings: VWABindings, device: str) -> None:
        self.bindings = bindings
        self.requested_device = device
        self._fn = None

    def get(self):
        if self._fn is None:
            import torch

            device = self.requested_device
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA evaluator was requested, but torch.cuda.is_available() is false"
                )
            dtype = torch.float16 if device == "cuda" else torch.float32
            self._fn = self.bindings.image_utils.get_captioning_fn(
                device, dtype, "Salesforce/blip2-flan-t5-xl"
            )
        return self._fn


def load_vwa_bindings(vwa_root: Path) -> VWABindings:
    """Import only the VWA environment/auth/evaluator APIs used by this runner."""
    if not vwa_root.is_dir():
        raise RuntimeError(f"VisualWebArena checkout not found: {vwa_root}")
    sys.path.insert(0, str(vwa_root))

    # VWA imports Hugging Face `evaluate` for StringSoftEvaluator, but its
    # evaluator_router cannot select that class. Config validation accepts only
    # the four routed VWA evaluators, so this unreachable optional dependency is
    # replaced with a fail-fast stub.
    if "evaluate" not in sys.modules:
        evaluate_stub = types.ModuleType("evaluate")

        def unsupported_evaluate(*_args, **_kwargs):
            raise RuntimeError("Unsupported VWA StringSoftEvaluator was requested")

        evaluate_stub.load = unsupported_evaluate
        sys.modules["evaluate"] = evaluate_stub

    from browser_env import (  # type: ignore[import-not-found]
        ActionTypes,
        ScriptBrowserEnv,
        create_key_press_action,
        create_keyboard_type_action,
        create_none_action,
        create_scroll_action,
        create_stop_action,
        create_go_back_action,
        create_go_forward_action,
    )
    from browser_env.auto_login import renew_comb
    from evaluation_harness import evaluator_router, image_utils

    return VWABindings(
        action_types=ActionTypes,
        script_browser_env=ScriptBrowserEnv,
        create_none_action=create_none_action,
        create_keyboard_type_action=create_keyboard_type_action,
        create_key_press_action=create_key_press_action,
        create_scroll_action=create_scroll_action,
        create_stop_action=create_stop_action,
        create_go_back_action=create_go_back_action,
        create_go_forward_action=create_go_forward_action,
        renew_comb=renew_comb,
        evaluator_router=evaluator_router,
        image_utils=image_utils,
    )


def ensure_auth(
    auth_dir: Path,
    domains: Iterable[str],
    refresh: bool,
    renew_comb: Any,
) -> None:
    """Create VWA login storage only when absent or explicitly refreshed."""
    auth_dir.mkdir(parents=True, exist_ok=True)
    for domain in domains:
        state_path = auth_dir / f"{domain}_state.json"
        if refresh or not state_path.exists():
            print(f"[auth] logging into {domain}", flush=True)
            renew_comb([domain], auth_folder=str(auth_dir))
        if not state_path.is_file():
            raise RuntimeError(f"Authentication state was not created: {state_path}")
