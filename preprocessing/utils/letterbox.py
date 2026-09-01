"""Redimensionamento proporcional e conversão de coordenadas."""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    frame: np.ndarray, infer_size: int
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Redimensiona sem distorcer e preenche as bordas com cinza neutro.

    Retorna a imagem, a escala uniforme e o padding esquerdo/superior.
    """
    if infer_size < 1:
        raise ValueError("infer_size deve ser positivo.")

    height, width = frame.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("O frame deve ter altura e largura positivas.")

    scale = min(infer_size / width, infer_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    total_pad_w = infer_size - resized_width
    total_pad_h = infer_size - resized_height
    pad_w = total_pad_w // 2
    pad_h = total_pad_h // 2
    return (
        cv2.copyMakeBorder(
            resized,
            pad_h,
            total_pad_h - pad_h,
            pad_w,
            total_pad_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        ),
        scale,
        (pad_w, pad_h),
    )


def adjust_bboxes(
    boxes_xyxy: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
    pad_w: int = 0,
    pad_h: int = 0,
) -> np.ndarray:
    """Converte caixas ``xyxy`` do frame processado ao frame original."""
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("As escalas devem ser positivas.")

    boxes = np.asarray(boxes_xyxy, dtype=float).copy()
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy deve ter shape (N, 4).")

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale_x
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale_y
    return boxes
