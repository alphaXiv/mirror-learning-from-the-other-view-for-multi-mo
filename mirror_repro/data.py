from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


BASE_URL = "https://raw.githubusercontent.com/lupantech/InterGPS/main/data/geometry3k"
ANSWER_LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Problem:
    problem_id: str
    question: str
    choices: tuple[str, ...]
    answer: str
    image_path: Path
    text_logic: tuple[str, ...]
    diagram_logic: tuple[str, ...]
    line_instances: tuple[str, ...]
    point_positions: dict[str, tuple[float, float]]


def _download(url: str, path: Path) -> None:
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".partial")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def prepare_geometry3k(root: Path) -> dict[str, Path]:
    """Download the public MIT-licensed InterGPS archives at run time."""
    root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for split in ("train", "val"):
        archive = root / f"{split}.zip"
        target = root / split
        _download(f"{BASE_URL}/{split}.zip", archive)
        if not target.exists():
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(root)
        outputs[split] = target
    return outputs


def _find_problem_dirs(split_root: Path) -> list[Path]:
    return sorted(
        (p.parent for p in split_root.rglob("data.json")),
        key=lambda p: int(p.name),
    )


def load_split(split_root: Path, limit: int, seed: int) -> list[Problem]:
    dirs = _find_problem_dirs(split_root)
    # A deterministic permutation prevents selecting only contiguous textbook pages.
    dirs.sort(key=lambda p: ((int(p.name) * 2654435761 + seed) % (2**32), int(p.name)))
    problems: list[Problem] = []
    for folder in dirs[:limit]:
        with (folder / "data.json").open(encoding="utf-8") as handle:
            row = json.load(handle)
        with (folder / "logic_form.json").open(encoding="utf-8") as handle:
            logic = json.load(handle)
        positions = {
            str(name): (float(xy[0]), float(xy[1]))
            for name, xy in logic.get("point_positions", {}).items()
            if isinstance(xy, list) and len(xy) == 2
        }
        problems.append(
            Problem(
                problem_id=str(row.get("id", folder.name)),
                question=str(row.get("annotat_text") or row["problem_text"]),
                choices=tuple(str(x) for x in row["choices"]),
                answer=str(row["answer"]).strip().upper(),
                image_path=folder / "img_diagram.png",
                text_logic=tuple(logic.get("text_logic_form", ())),
                diagram_logic=tuple(logic.get("diagram_logic_form", ())),
                line_instances=tuple(logic.get("line_instances", ())),
                point_positions=positions,
            )
        )
    return problems


def renderer_audit(problems: list[Problem]) -> dict[str, float | int]:
    """Check whether released structured geometry can account for rendered lines."""
    total_lines = 0
    resolvable_lines = 0
    valid_images = 0
    for problem in problems:
        try:
            with Image.open(problem.image_path) as image:
                valid_images += int(image.width > 0 and image.height > 0)
        except OSError:
            pass
        for line in problem.line_instances:
            total_lines += 1
            if len(line) == 2 and line[0] in problem.point_positions and line[1] in problem.point_positions:
                resolvable_lines += 1
    return {
        "problems": len(problems),
        "valid_image_fraction": valid_images / max(1, len(problems)),
        "structured_lines": total_lines,
        "resolvable_line_fraction": resolvable_lines / max(1, total_lines),
    }


def open_image(problem: Problem, max_side: int) -> Image.Image:
    image = Image.open(problem.image_path).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _choice_block(problem: Problem) -> str:
    return "\n".join(
        f"({ANSWER_LETTERS[i]}) {choice}" for i, choice in enumerate(problem.choices)
    )


def view_payload(problem: Problem, view: str, max_side: int) -> tuple[str, Image.Image | None]:
    shared = (
        f"{problem.question}\n\nChoices:\n{_choice_block(problem)}\n\n"
        "Return exactly one line in the form `Final answer: X`, where X is A, B, C, or D."
    )
    facts = "\n".join((*problem.text_logic, *problem.diagram_logic))
    facts = facts[:12000]
    if view == "text":
        return (
            "Solve the geometry problem from its complete structured textual view. "
            "The formal facts below encode the diagram without pixels.\n\n"
            f"{shared}\n\nFormal geometry facts:\n{facts}",
            None,
        )
    if view == "image":
        return (
            "Solve the geometry problem from the diagram and accompanying question.\n\n" + shared,
            open_image(problem, max_side),
        )
    if view == "combined":
        return (
            "Solve using both the diagram and the structured textual facts.\n\n"
            f"{shared}\n\nFormal geometry facts:\n{facts}",
            open_image(problem, max_side),
        )
    raise ValueError(f"Unknown view: {view}")


def shared_data_root() -> Path:
    return Path(os.environ.get("MIRROR_DATA_ROOT", "/tmp/mirror-geometry3k"))

