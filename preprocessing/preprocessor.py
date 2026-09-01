"""Módulo central de pré-processamento de imagens para inferência YOLO."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from preprocessing.utils.letterbox import adjust_bboxes, letterbox


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuração imutável aplicada igualmente a cada frame."""

    infer_size: int = 320
    convert_rgb: bool = True
    use_letterbox: bool = True
    gaussian_blur: bool = False
    gaussian_ksize: int = 3
    gaussian_sigma: float = 0.8
    median_blur: bool = False
    median_ksize: int = 3
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    clahe_space: str = "lab"
    normalize: bool = False


@dataclass(frozen=True)
class PreprocessResult:
    """Frame transformado e metadados para reverter bounding boxes."""

    frame: np.ndarray
    scale: float = 1.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    pad_w: int = 0
    pad_h: int = 0
    orig_size: tuple[int, int] = (0, 0)


class Preprocessor:
    """Pipeline puro e reutilizável para frames BGR do OpenCV."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.cfg = config or PreprocessConfig()
        self._validate_config()
        self._clahe = (
            cv2.createCLAHE(
                clipLimit=self.cfg.clahe_clip,
                tileGridSize=(self.cfg.clahe_tile, self.cfg.clahe_tile),
            )
            if self.cfg.clahe
            else None
        )

    def process(self, frame: np.ndarray) -> PreprocessResult:
        """Aplica as transformações ao frame BGR, sem modificar a entrada."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("O frame deve ter shape (altura, largura, 3).")

        orig_h, orig_w = frame.shape[:2]
        out = frame.copy()
        if self.cfg.clahe:
            out = self._apply_clahe(out)
        if self.cfg.convert_rgb:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        if self.cfg.gaussian_blur:
            out = cv2.GaussianBlur(
                out,
                (self.cfg.gaussian_ksize, self.cfg.gaussian_ksize),
                sigmaX=self.cfg.gaussian_sigma,
            )
        elif self.cfg.median_blur:
            out = cv2.medianBlur(out, self.cfg.median_ksize)

        if self.cfg.use_letterbox:
            out, scale, (pad_w, pad_h) = letterbox(out, self.cfg.infer_size)
            scale_x = scale_y = scale
        else:
            out = cv2.resize(out, (self.cfg.infer_size, self.cfg.infer_size))
            scale_x = self.cfg.infer_size / orig_w
            scale_y = self.cfg.infer_size / orig_h
            scale = min(scale_x, scale_y)
            pad_w = pad_h = 0

        if self.cfg.normalize:
            out = out.astype(np.float32) / 255.0

        return PreprocessResult(
            frame=out,
            scale=scale,
            scale_x=scale_x,
            scale_y=scale_y,
            pad_w=pad_w,
            pad_h=pad_h,
            orig_size=(orig_h, orig_w),
        )

    def adjust_boxes(
        self, boxes_xyxy: np.ndarray, result: PreprocessResult
    ) -> np.ndarray:
        """Traz caixas do espaço processado de volta ao frame original."""
        return adjust_bboxes(
            boxes_xyxy,
            scale_x=result.scale_x,
            scale_y=result.scale_y,
            pad_w=result.pad_w,
            pad_h=result.pad_h,
        )

    def _validate_config(self) -> None:
        if self.cfg.infer_size < 1:
            raise ValueError("infer_size deve ser positivo.")
        for name, kernel in (
            ("gaussian_ksize", self.cfg.gaussian_ksize),
            ("median_ksize", self.cfg.median_ksize),
        ):
            if kernel < 1 or kernel % 2 == 0:
                raise ValueError(f"{name} deve ser ímpar e positivo.")
        if self.cfg.clahe_tile < 1 or self.cfg.clahe_clip <= 0:
            raise ValueError("Os parâmetros do CLAHE devem ser positivos.")
        if self.cfg.clahe_space not in {"lab", "hsv"}:
            raise ValueError("clahe_space deve ser 'lab' ou 'hsv'.")

    def _apply_clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        assert self._clahe is not None
        if self.cfg.clahe_space == "lab":
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            lightness, a_channel, b_channel = cv2.split(lab)
            equalized = self._clahe.apply(lightness)
            return cv2.cvtColor(
                cv2.merge([equalized, a_channel, b_channel]), cv2.COLOR_LAB2BGR
            )

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        equalized = self._clahe.apply(value)
        return cv2.cvtColor(
            cv2.merge([hue, saturation, equalized]), cv2.COLOR_HSV2BGR
        )


CONFIG_DEFAULT = PreprocessConfig()
CONFIG_LOW_LIGHT = PreprocessConfig(clahe=True, clahe_clip=2.0, clahe_tile=8)
CONFIG_HIGH_QUALITY = PreprocessConfig(infer_size=640)
