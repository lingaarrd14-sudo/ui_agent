"""VisualWebArena 사이트 환경과 task config를 읽기 전용 원본에서 준비한다."""

from __future__ import annotations

import io
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from PIL import Image


UI_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = UI_ROOT.parent
VWA_ROOT = WORKSPACE_ROOT / "visualwebarena"

SITE_DEFAULTS = {
    "CLASSIFIEDS": "http://127.0.0.1:9980",
    "SHOPPING": "http://127.0.0.1:7770",
    "REDDIT": "http://127.0.0.1:9999",
    "WIKIPEDIA": "http://127.0.0.1:8888",
    "HOMEPAGE": "http://127.0.0.1:4399",
}
CLASSIFIEDS_RESET_TOKEN = "4b61655535e7ed388f0d40a93600254c"
DOMAIN_SOURCES = {
    "classifieds": "test_classifieds.raw.json",
    "reddit": "test_reddit.raw.json",
    "shopping": "test_shopping.raw.json",
}
SUPPORTED_EVAL_TYPES = {
    "string_match",
    "url_match",
    "program_html",
    "page_image_query",
}


def load_local_environment() -> None:
    """Load the UI-agent .env without overriding exported shell variables."""
    load_dotenv(UI_ROOT / ".env")


def configure_environment(require_api_key: bool) -> tuple[str | None, str | None]:
    """Establish VWA's documented site variables and return model credentials."""
    load_local_environment()
    for name, value in SITE_DEFAULTS.items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("DATASET", "visualwebarena")
    os.environ.setdefault("CLASSIFIEDS_RESET_TOKEN", CLASSIFIEDS_RESET_TOKEN)

    # Keep direct VWA helper subprocesses able to resolve browser_env.
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    python_paths = current_pythonpath.split(os.pathsep) if current_pythonpath else []
    if str(VWA_ROOT) not in python_paths:
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(VWA_ROOT), *python_paths]
        ).rstrip(os.pathsep)

    api_key = os.environ.get("OPENAI_API_KEY")
    if require_api_key and not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured in ui_agent/.env")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL")
    return api_key, base_url


def site_urls() -> dict[str, str]:
    return {name: os.environ[name] for name in SITE_DEFAULTS}


def replace_placeholders(value: Any) -> Any:
    """Resolve VWA website placeholders recursively without changing intent text."""
    if isinstance(value, str):
        for name, url in site_urls().items():
            value = value.replace(f"__{name}__", url)
        return value
    if isinstance(value, list):
        return [replace_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_placeholders(item) for key, item in value.items()}
    return value


def generate_configs(result_dir: Path, domains: Sequence[str]) -> dict[str, list[Path]]:
    """Materialize executable configs outside the VWA checkout."""
    generated: dict[str, list[Path]] = {}
    for domain in domains:
        source = VWA_ROOT / "config_files" / "vwa" / DOMAIN_SOURCES[domain]
        tasks = replace_placeholders(json.loads(source.read_text(encoding="utf-8")))
        output_dir = result_dir / "configs" / domain
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for task in tasks:
            image_spec = task.get("image")
            if image_spec:
                values = [image_spec] if isinstance(image_spec, str) else image_spec
                resolved = [
                    value
                    if value.startswith(("http://", "https://"))
                    else str((VWA_ROOT / value).resolve())
                    for value in values
                ]
                task["image"] = resolved[0] if isinstance(image_spec, str) else resolved
            path = output_dir / f"{task['task_id']}.json"
            path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            paths.append(path)
        generated[domain] = paths
    return generated


def task_selection(
    generated: dict[str, list[Path]], start: int, end: int | None
) -> list[tuple[str, Path]]:
    """Apply the same half-open index range independently to each domain."""
    selected: list[tuple[str, Path]] = []
    for domain, paths in generated.items():
        upper = len(paths) if end is None else min(end, len(paths))
        selected.extend((domain, path) for path in paths[start:upper])
    return selected


def validate_selection(selected: Sequence[tuple[str, Path]]) -> dict[str, int]:
    """Fail before browser/API use when task contracts or reference files are invalid."""
    if not selected:
        raise ValueError("No tasks were selected; check --domain/--start/--end")
    counts = {"tasks": len(selected), "reference_images": 0, "requires_reset": 0}
    required = {"task_id", "intent", "start_url", "eval"}
    for domain, path in selected:
        task = json.loads(path.read_text(encoding="utf-8"))
        missing = required - task.keys()
        if missing:
            raise ValueError(f"{domain}/{path.name} is missing {sorted(missing)}")
        eval_types = set(task["eval"].get("eval_types", []))
        unsupported = eval_types - SUPPORTED_EVAL_TYPES
        if unsupported:
            raise ValueError(f"{domain}/{path.name}: unsupported eval types {unsupported}")
        if task.get("require_reset"):
            counts["requires_reset"] += 1
        image_spec = task.get("image")
        image_paths = [] if not image_spec else (
            [image_spec] if isinstance(image_spec, str) else image_spec
        )
        counts["reference_images"] += len(image_paths)
        for image_path in image_paths:
            if not image_path.startswith(("http://", "https://")) and not Path(
                image_path
            ).is_file():
                raise ValueError(f"Missing reference image: {image_path}")
    return counts


def load_reference_images(image_spec: str | list[str] | None) -> list[Image.Image]:
    """Load task-supplied reference images without generating captions."""
    if not image_spec:
        return []
    paths = [image_spec] if isinstance(image_spec, str) else image_spec
    images: list[Image.Image] = []
    for path in paths:
        if path.startswith(("http://", "https://")):
            response = requests.get(path, timeout=30)
            response.raise_for_status()
            source: Any = io.BytesIO(response.content)
        else:
            source = path
        with Image.open(source) as image:
            images.append(image.convert("RGB").copy())
    return images
