"""Métricas mAP@0.5 e cliente mínimo para inferência Roboflow."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    image_id: str
    class_id: int
    confidence: float
    box: np.ndarray


def iou_xyxy(first: np.ndarray, second: np.ndarray) -> float:
    """Calcula IoU para duas caixas ``[x1, y1, x2, y2]``."""
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def average_precision(
    predictions: list[Detection], ground_truth: dict[str, dict[int, list[np.ndarray]]], class_id: int
) -> float:
    """AP com interpolação de 101 pontos, em IoU 0,5."""
    total = sum(len(image_boxes.get(class_id, [])) for image_boxes in ground_truth.values())
    if not total:
        return 0.0

    matched = {image_id: np.zeros(len(boxes.get(class_id, [])), dtype=bool) for image_id, boxes in ground_truth.items()}
    ordered = sorted((item for item in predictions if item.class_id == class_id), key=lambda item: item.confidence, reverse=True)
    true_positive = np.zeros(len(ordered), dtype=float)
    false_positive = np.zeros(len(ordered), dtype=float)
    for index, prediction in enumerate(ordered):
        candidates = ground_truth.get(prediction.image_id, {}).get(class_id, [])
        if not candidates:
            false_positive[index] = 1
            continue
        overlaps = [iou_xyxy(prediction.box, candidate) for candidate in candidates]
        best = int(np.argmax(overlaps))
        if overlaps[best] >= 0.5 and not matched[prediction.image_id][best]:
            true_positive[index] = 1
            matched[prediction.image_id][best] = True
        else:
            false_positive[index] = 1

    recall = np.cumsum(true_positive) / total
    precision = np.cumsum(true_positive) / np.maximum(np.cumsum(true_positive) + np.cumsum(false_positive), 1)
    return float(np.mean([max(precision[recall >= level], default=0.0) for level in np.linspace(0, 1, 101)]))


def map50(predictions: list[Detection], ground_truth: dict[str, dict[int, list[np.ndarray]]], class_count: int) -> float:
    """Média de AP@0,5 entre todas classes do dataset."""
    return float(np.mean([average_precision(predictions, ground_truth, class_id) for class_id in range(class_count)]))


def read_yolo_boxes(label_path: Path, width: int, height: int) -> dict[int, list[np.ndarray]]:
    """Lê labels YOLO normalizados e devolve caixas em pixels."""
    result: dict[int, list[np.ndarray]] = {}
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        class_raw, center_x, center_y, box_width, box_height = raw_line.split()
        class_id = int(class_raw)
        center_x, center_y, box_width, box_height = (float(value) for value in (center_x, center_y, box_width, box_height))
        result.setdefault(class_id, []).append(
            np.array(
                [
                    (center_x - box_width / 2) * width,
                    (center_y - box_height / 2) * height,
                    (center_x + box_width / 2) * width,
                    (center_y + box_height / 2) * height,
                ],
                dtype=float,
            )
        )
    return result


def infer(endpoint: str, image: np.ndarray, class_names: list[str]) -> list[Detection]:
    """Envia imagem ao endpoint Serverless e retorna detecções normalizadas."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY não está definida.")
    encoded_ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not encoded_ok:
        raise RuntimeError("Não foi possível codificar imagem para inferência.")
    query = urlencode({"api_key": api_key, "confidence": 1, "overlap": 100})
    request = Request(
        f"{endpoint}?{query}",
        data=base64.b64encode(encoded.tobytes()),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:  # nosec B310: endpoint definido pelo projeto
        payload = json.loads(response.read().decode("utf-8"))

    lookup = {name.casefold(): index for index, name in enumerate(class_names)}
    detections: list[Detection] = []
    for item in payload.get("predictions", []):
        class_id = lookup.get(str(item["class"]).casefold())
        if class_id is None:
            continue
        width = float(item["width"])
        height = float(item["height"])
        center_x = float(item["x"])
        center_y = float(item["y"])
        detections.append(
            Detection(
                image_id="",
                class_id=class_id,
                confidence=float(item["confidence"]),
                box=np.array(
                    [center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2],
                    dtype=float,
                ),
            )
        )
    return detections
