"""Gera um dataset ImageFolder de classificação a partir das anotações YOLO EPI.

Uso:
    python scripts/prepare_aula6_classification_dataset.py

O resultado possui ``train/<classe>`` e ``validation/<classe>`` e pode ser
carregado diretamente por TensorFlow/Keras ou torchvision.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SPLITS = {"train": "train", "valid": "validation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/epi-v1"))
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/aula6-classification")
    )
    parser.add_argument("--padding", type=float, default=0.08)
    return parser.parse_args()


def class_names(source: Path) -> list[str]:
    metadata = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    names = metadata.get("names") if isinstance(metadata, dict) else None
    if not isinstance(names, list) or len(names) < 2:
        raise ValueError("data.yaml precisa declarar ao menos duas classes em names.")
    return [str(name) for name in names]


def crop_box(
    image: cv2.typing.MatLike,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    padding: float,
) -> cv2.typing.MatLike | None:
    image_height, image_width = image.shape[:2]
    x1 = max(0, int((center_x - width / 2 - padding * width) * image_width))
    y1 = max(0, int((center_y - height / 2 - padding * height) * image_height))
    x2 = min(image_width, int((center_x + width / 2 + padding * width) * image_width))
    y2 = min(image_height, int((center_y + height / 2 + padding * height) * image_height))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return image[y1:y2, x1:x2]


def write_crops(
    source: Path, output: Path, names: list[str], padding: float
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for source_split, target_split in SPLITS.items():
        images_dir = source / source_split / "images"
        labels_dir = source / source_split / "labels"
        for class_name in names:
            (output / target_split / class_name).mkdir(parents=True, exist_ok=True)

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Imagem inválida: {image_path}")
            for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Rótulo inválido em {label_path}: {line!r}")
                class_id = int(fields[0])
                if not 0 <= class_id < len(names):
                    raise ValueError(f"Classe inválida em {label_path}: {class_id}")
                box = [float(value) for value in fields[1:]]
                crop = crop_box(image, *box, padding=padding)
                if crop is None:
                    continue
                target = output / target_split / names[class_id] / f"{image_path.stem}_{index}.jpg"
                if not cv2.imwrite(str(target), crop):
                    raise OSError(f"Não foi possível escrever {target}")
                counts[f"{target_split}/{names[class_id]}"] += 1
    return counts


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not 0 <= args.padding < 0.5:
        raise SystemExit("--padding deve estar entre 0 e 0.5.")
    if not source.is_dir():
        raise SystemExit(f"Dataset de origem não encontrado: {source}")
    if output.exists():
        raise SystemExit(f"Saída já existe; não sobrescrevo dados: {output}")

    names = class_names(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        counts = write_crops(source, output, names, args.padding)
        missing = [name for name in names if not counts[f"train/{name}"] or not counts[f"validation/{name}"]]
        if missing:
            raise ValueError(f"Classes sem recortes em train ou validation: {missing}")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise

    print(f"Dataset criado em: {output}")
    for name in names:
        print(f"train/{name}: {counts[f'train/{name}']}")
        print(f"validation/{name}: {counts[f'validation/{name}']}")


if __name__ == "__main__":
    main()
