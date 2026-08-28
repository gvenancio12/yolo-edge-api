"""Converte Construction-PPE para o dataset EPI v1 em formato YOLO.

O conjunto de origem é público e deve ser baixado separadamente. Este script retém
apenas as classes helmet, vest e Person e as renomeia para português.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SOURCE_CLASS_MAP = {0: 0, 2: 1, 6: 2}
TARGET_CLASSES = ["capacete", "colete", "pessoa"]
SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
SOURCE_LICENSE = "AGPL-3.0"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class Candidate:
    image_path: Path
    labels: list[str]
    source_split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def remap_labels(label_path: Path) -> list[str]:
    remapped: list[str] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = raw_line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number} não é um rótulo YOLO válido.")
        source_class = int(values[0])
        if source_class in SOURCE_CLASS_MAP:
            remapped.append(" ".join([str(SOURCE_CLASS_MAP[source_class]), *values[1:]]))
    return remapped


def collect_candidates(source: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    for split in ("train", "val", "test"):
        images_dir = source / "images" / split
        labels_dir = source / "labels" / split
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Estrutura esperada ausente: {images_dir} e {labels_dir}."
            )
        for image_path in images_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            labels = remap_labels(label_path)
            if labels:
                candidates.append(Candidate(image_path, labels, split))
    return candidates


def split_counts(count: int) -> dict[str, int]:
    train = round(count * 0.70)
    valid = round(count * 0.15)
    return {"train": train, "valid": valid, "test": count - train - valid}


def prepare_directory(target: Path) -> None:
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            f"O destino {target} já contém arquivos; use um diretório vazio."
        )
    for split in ("train", "valid", "test"):
        (target / split / "images").mkdir(parents=True, exist_ok=True)
        (target / split / "labels").mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(target: Path) -> None:
    content = "\n".join(
        [
            "path: .",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "names:",
            "  0: capacete",
            "  1: colete",
            "  2: pessoa",
            "",
        ]
    )
    (target / "data.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.count < 500:
        raise SystemExit("Use --count >= 500 para garantir 350 imagens de treino.")
    source = args.source.resolve()
    target = args.target.resolve()
    candidates = collect_candidates(source)
    if len(candidates) < args.count:
        raise SystemExit(
            f"A fonte possui apenas {len(candidates)} imagens com classes EPI; "
            f"a meta é {args.count}."
        )

    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: args.count]
    counts = split_counts(args.count)
    prepare_directory(target)
    class_annotations: Counter[int] = Counter()
    source_splits: Counter[str] = Counter()
    offset = 0
    for split, count in counts.items():
        for index, candidate in enumerate(selected[offset : offset + count], 1):
            name = f"epi_{split}_{index:04d}"
            image_target = target / split / "images" / f"{name}{candidate.image_path.suffix.lower()}"
            label_target = target / split / "labels" / f"{name}.txt"
            shutil.copy2(candidate.image_path, image_target)
            label_target.write_text("\n".join(candidate.labels) + "\n", encoding="utf-8")
            source_splits[candidate.source_split] += 1
            for label in candidate.labels:
                class_annotations[int(label.split()[0])] += 1
        offset += count

    write_dataset_yaml(target)
    provenance = "\n".join(
        [
            "# Proveniência do Dataset EPI v1",
            "",
            "Fonte: Ultralytics Construction-PPE.",
            f"URL: {SOURCE_URL}",
            f"Licença: {SOURCE_LICENSE}.",
            "",
            "Amostra determinística de 500 imagens; rótulos retidos e renomeados:",
            "helmet -> capacete, vest -> colete, Person -> pessoa.",
        ]
    )
    (target / "SOURCE.md").write_text(provenance, encoding="utf-8")
    summary = {
        "source": str(source),
        "target": str(target),
        "source_url": SOURCE_URL,
        "license": SOURCE_LICENSE,
        "seed": args.seed,
        "counts": counts,
        "source_splits": dict(source_splits),
        "class_annotations": {
            TARGET_CLASSES[index]: class_annotations[index]
            for index in range(len(TARGET_CLASSES))
        },
    }
    (target / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
