"""Prepara um subconjunto EPI v1, em formato YOLO, de uma fonte pública."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TARGET_CLASSES = ["capacete", "colete", "pessoa"]
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
    parser.add_argument("--source-map", default="0:0,2:1,6:2")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-license", required=True)
    return parser.parse_args()


def parse_source_map(value: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for item in value.split(","):
        try:
            source, target = (int(part) for part in item.split(":", 1))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "--source-map deve usar pares como 1:0,0:1,11:2."
            ) from error
        if target not in range(len(TARGET_CLASSES)):
            raise argparse.ArgumentTypeError("Classe EPI de destino inválida.")
        mapping[source] = target
    if set(mapping.values()) != set(range(len(TARGET_CLASSES))):
        raise argparse.ArgumentTypeError(
            "--source-map deve cobrir capacete, colete e pessoa."
        )
    return mapping


def remap_labels(label_path: Path, source_class_map: dict[int, int]) -> list[str]:
    remapped: list[str] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = raw_line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number} não é um rótulo YOLO válido.")
        source_class = int(values[0])
        if source_class in source_class_map:
            remapped.append(" ".join([str(source_class_map[source_class]), *values[1:]]))
    return remapped


def find_split_directories(source: Path, split: str) -> tuple[Path, Path]:
    for images_dir, labels_dir in (
        (source / "images" / split, source / "labels" / split),
        (source / split / "images", source / split / "labels"),
    ):
        if images_dir.is_dir() and labels_dir.is_dir():
            return images_dir, labels_dir
    raise FileNotFoundError(f"Estrutura esperada ausente para {split} em {source}.")


def collect_candidates(source: Path, source_class_map: dict[int, int]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for split in ("train", "val", "test"):
        images_dir, labels_dir = find_split_directories(source, split)
        for image_path in images_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            labels = remap_labels(label_path, source_class_map)
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
    source_class_map = parse_source_map(args.source_map)
    candidates = collect_candidates(source, source_class_map)
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
            f"Fonte: {args.source_name}.",
            f"URL: {args.source_url}",
            f"Licença: {args.source_license}.",
            "",
            "Amostra determinística de 500 imagens; rótulos retidos e renomeados:",
            f"Mapeamento de classes de origem: {args.source_map}.",
        ]
    )
    (target / "SOURCE.md").write_text(provenance, encoding="utf-8")
    summary = {
        "source": str(source),
        "target": str(target),
        "source_name": args.source_name,
        "source_url": args.source_url,
        "license": args.source_license,
        "source_map": source_class_map,
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
