"""Executa E1-A até E4-C usando validação local e inferência Roboflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from preprocessing.utils.letterbox import letterbox
from preprocessing.utils.roboflow_evaluate import Detection, infer, map50, read_yolo_boxes

DEFAULT_ENDPOINT = "https://serverless.roboflow.com/epi-detection-rpi5-6q1f3-gk9hv/1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/epi-v1"))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--report", type=Path, default=Path("evidencias-aula5/resultados.json"))
    return parser.parse_args()


def global_equalization(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)


def clahe(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge([equalized, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def transform_boxes(boxes: dict[int, list[np.ndarray]], scale_x: float, scale_y: float, pad_x: int = 0, pad_y: int = 0) -> dict[int, list[np.ndarray]]:
    transformed: dict[int, list[np.ndarray]] = {}
    for class_id, items in boxes.items():
        transformed[class_id] = []
        for box in items:
            output = box.copy()
            output[[0, 2]] = output[[0, 2]] * scale_x + pad_x
            output[[1, 3]] = output[[1, 3]] * scale_y + pad_y
            transformed[class_id].append(output)
    return transformed


def variants() -> list[tuple[str, str]]:
    return [
        ("E1-A", "BGR sem conversão"),
        ("E1-B", "RGB correto"),
        ("E2-A", "resize simples 320x320"),
        ("E2-B", "letterbox 320, padding 114"),
        ("E3-A", "sem filtro"),
        ("E3-B", "GaussianBlur 3x3, sigma=0.8"),
        ("E3-C", "GaussianBlur 5x5, sigma=1.5"),
        ("E3-D", "medianBlur kernel=3"),
        ("E4-A", "escurecida, sem equalização"),
        ("E4-B", "escurecida, equalizeHist global"),
        ("E4-C", "escurecida, CLAHE clip=2 tile=8"),
    ]


def process_variant(name: str, frame: np.ndarray, boxes: dict[int, list[np.ndarray]]) -> tuple[np.ndarray, dict[int, list[np.ndarray]]]:
    height, width = frame.shape[:2]
    if name == "E1-A":
        return frame[:, :, ::-1], boxes
    if name in {"E1-B", "E3-A"}:
        return frame, boxes
    if name == "E2-A":
        return cv2.resize(frame, (320, 320)), transform_boxes(boxes, 320 / width, 320 / height)
    if name == "E2-B":
        output, scale, (pad_x, pad_y) = letterbox(frame, 320)
        return output, transform_boxes(boxes, scale, scale, pad_x, pad_y)
    if name == "E3-B":
        return cv2.GaussianBlur(frame, (3, 3), sigmaX=0.8), boxes
    if name == "E3-C":
        return cv2.GaussianBlur(frame, (5, 5), sigmaX=1.5), boxes
    if name == "E3-D":
        return cv2.medianBlur(frame, 3), boxes

    dark = cv2.convertScaleAbs(frame, alpha=0.35, beta=0)
    if name == "E4-A":
        return dark, boxes
    if name == "E4-B":
        return global_equalization(dark), boxes
    if name == "E4-C":
        return clahe(dark), boxes
    raise ValueError(f"Experimento desconhecido: {name}")


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    metadata = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    class_names = [str(name) for name in metadata["names"]]
    image_paths = sorted((dataset / "valid" / "images").glob("*.jpg"))
    if not image_paths:
        raise SystemExit("Nenhuma imagem encontrada em valid/images.")

    result_rows = []
    baseline = None
    for experiment, configuration in variants():
        predictions: list[Detection] = []
        truth: dict[str, dict[int, list[np.ndarray]]] = {}
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise SystemExit(f"Não foi possível abrir {image_path}.")
            boxes = read_yolo_boxes(image_path.parent.parent / "labels" / f"{image_path.stem}.txt", frame.shape[1], frame.shape[0])
            transformed, transformed_boxes = process_variant(experiment, frame, boxes)
            truth[image_path.name] = transformed_boxes
            predictions.extend(replace(item, image_id=image_path.name) for item in infer(args.endpoint, transformed, class_names))

        value = map50(predictions, truth, len(class_names))
        if baseline is None:
            baseline = value
        result_rows.append(
            {
                "experiment": experiment,
                "configuration": configuration,
                "map50": round(value, 6),
                "delta_vs_baseline": round(value - baseline, 6),
                "images": len(image_paths),
            }
        )
        print(f"{experiment}: mAP@0.5={value:.4f}; delta={value - baseline:+.4f}", flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "model_endpoint": args.endpoint,
                "split": "valid",
                "classes": class_names,
                "baseline": "E1-A",
                "results": result_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
