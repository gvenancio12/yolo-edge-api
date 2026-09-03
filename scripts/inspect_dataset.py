"""Valida a estrutura e os rótulos de um dataset YOLO de EPI.

Exemplo:
    python3 scripts/inspect_dataset.py --dataset datasets/epi-v1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
EXPECTED_SPLITS = {"train": 0.70, "valid": 0.15, "test": 0.15}
EXPECTED_CLASSES = ["capacete", "colete", "pessoa"]


@dataclass
class SplitReport:
    images: int = 0
    labels: int = 0
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--min-train",
        type=int,
        default=0,
        help="Mínimo de imagens de treino; use 0 para não exigir mínimo.",
    )
    parser.add_argument("--split-tolerance", type=float, default=0.03)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def image_stems(directory: Path) -> set[str]:
    return {
        image.stem
        for image in directory.iterdir()
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    }


def label_stems(directory: Path) -> set[str]:
    return {label.stem for label in directory.glob("*.txt") if label.is_file()}


def validate_label(
    label_path: Path, class_count: int, errors: list[str], class_counts: Counter[int]
) -> None:
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = raw_line.split()
        if len(values) != 5:
            errors.append(f"{label_path}:{line_number} deve ter 5 valores YOLO.")
            continue
        try:
            class_id = int(values[0])
            x, y, width, height = (float(value) for value in values[1:])
        except ValueError:
            errors.append(f"{label_path}:{line_number} contém valores inválidos.")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"{label_path}:{line_number} possui classe fora do intervalo.")
        else:
            class_counts[class_id] += 1
        if not 0 <= x <= 1 or not 0 <= y <= 1 or not 0 < width <= 1 or not 0 < height <= 1:
            errors.append(f"{label_path}:{line_number} possui bounding box fora de [0, 1].")


def normalize_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        return [str(raw_names[index]) for index in sorted(raw_names, key=int)]
    raise ValueError("data.yaml deve declarar names como lista ou mapa de índices.")


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    if args.min_train < 0:
        raise SystemExit("--min-train não pode ser negativo.")
    errors: list[str] = []
    reports: dict[str, SplitReport] = {}
    class_counts: Counter[int] = Counter()
    data_yaml = dataset / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"REPROVADO: data.yaml ausente em {dataset}")

    metadata = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    try:
        names = normalize_names(metadata.get("names"))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"REPROVADO: {exc}") from exc
    if names != EXPECTED_CLASSES:
        errors.append(
            f"classes esperadas={EXPECTED_CLASSES}; encontradas={names}."
        )

    for split in EXPECTED_SPLITS:
        images_dir = dataset / split / "images"
        labels_dir = dataset / split / "labels"
        report = SplitReport()
        reports[split] = report
        if not images_dir.is_dir() or not labels_dir.is_dir():
            errors.append(f"split '{split}' precisa conter images/ e labels/.")
            continue
        images = image_stems(images_dir)
        labels = label_stems(labels_dir)
        report.images = len(images)
        report.labels = len(labels)
        report.missing_labels = sorted(images - labels)
        report.orphan_labels = sorted(labels - images)
        if report.missing_labels:
            errors.append(f"{split}: {len(report.missing_labels)} imagens sem rótulo.")
        if report.orphan_labels:
            errors.append(f"{split}: {len(report.orphan_labels)} rótulos sem imagem.")
        for label_path in labels_dir.glob("*.txt"):
            validate_label(label_path, len(names), errors, class_counts)

    total_images = sum(report.images for report in reports.values())
    if args.min_train and reports["train"].images < args.min_train:
        errors.append(
            f"train possui {reports['train'].images}; mínimo exigido={args.min_train}."
        )
    if total_images == 0:
        errors.append("dataset não contém imagens.")
    else:
        for split, expected_ratio in EXPECTED_SPLITS.items():
            actual_ratio = reports[split].images / total_images
            if abs(actual_ratio - expected_ratio) > args.split_tolerance:
                errors.append(
                    f"split {split}={actual_ratio:.1%}; esperado={expected_ratio:.0%}."
                )
    missing_classes = [
        names[index] for index in range(len(names)) if class_counts[index] == 0
    ]
    if missing_classes:
        errors.append(f"classes sem anotações: {missing_classes}.")

    result = {
        "approved": not errors,
        "dataset": str(dataset),
        "classes": names,
        "splits": {name: asdict(report) for name, report in reports.items()},
        "total_images": total_images,
        "class_annotations": {names[index]: class_counts[index] for index in range(len(names))},
        "errors": errors,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit("REPROVADO: corrija os erros acima.")
    print("APROVADO: estrutura, splits, classes e rótulos do dataset validados.")


if __name__ == "__main__":
    main()
