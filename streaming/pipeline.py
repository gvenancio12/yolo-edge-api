"""Componentes compartilhados pelos experimentos de streaming na Raspberry Pi."""

from __future__ import annotations

import base64
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_API_URL = "http://127.0.0.1:8000/predict"
DEFAULT_SIZE = (640, 480)


class InferenceError(RuntimeError):
    """Falha de comunicação ou de resposta da API de inferência."""


@dataclass(frozen=True)
class StreamConfig:
    api_url: str = DEFAULT_API_URL
    confidence: float = 0.25
    width: int = DEFAULT_SIZE[0]
    height: int = DEFAULT_SIZE[1]
    jpeg_quality: int = 80
    api_timeout_s: float = 30.0


@dataclass
class InferenceResult:
    detections: list[dict[str, Any]]
    api_ms: float
    inference_ms: float | None
    error: str | None = None


@dataclass
class FrameMetrics:
    frame: int
    capture_ms: float
    encode_ms: float
    api_ms: float
    inference_ms: float | None
    total_ms: float
    detections: int
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Camera:
    """Adaptador fino para a câmera CSI via Picamera2."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._camera: Any | None = None

    def __enter__(self) -> Camera:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 não está disponível. Execute este módulo na Raspberry Pi."
            ) from exc

        self._camera = Picamera2()
        config = self._camera.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        self._camera.configure(config)
        self._camera.start()
        time.sleep(0.8)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera.close()
            self._camera = None

    def capture(self) -> tuple[np.ndarray, float]:
        """Captura uma imagem BGR e retorna sua latência em milissegundos."""
        if self._camera is None:
            raise RuntimeError("A câmera não foi iniciada.")
        started = time.perf_counter()
        rgb = self._camera.capture_array("main")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr, (time.perf_counter() - started) * 1000


class ApiInferenceClient:
    """Cliente sem dependências extras para o endpoint /predict da API YOLO."""

    def __init__(self, config: StreamConfig) -> None:
        self.config = config

    def infer(self, frame_bgr: np.ndarray) -> tuple[InferenceResult, float]:
        encode_started = time.perf_counter()
        encoded, jpeg = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
        )
        encode_ms = (time.perf_counter() - encode_started) * 1000
        if not encoded:
            raise InferenceError("Falha ao codificar o frame em JPEG.")

        payload = json.dumps(
            {
                "image_base64": base64.b64encode(jpeg.tobytes()).decode("ascii"),
                "confidence": self.config.confidence,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.config.api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.api_timeout_s
            ) as response:
                body = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return InferenceResult([], elapsed_ms, None, str(exc)), encode_ms

        elapsed_ms = (time.perf_counter() - started) * 1000
        return (
            InferenceResult(
                detections=body.get("detections", []),
                api_ms=elapsed_ms,
                inference_ms=body.get("inference_ms"),
            ),
            encode_ms,
        )


def annotate_frame(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    stream_fps: float,
    inference: InferenceResult | None,
    inference_age_s: float | None,
) -> np.ndarray:
    """Desenha bounding boxes e OSD sem modificar o buffer compartilhado da câmera."""
    annotated = frame_bgr.copy()
    for detection in detections:
        bbox = detection.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(value) for value in bbox)
        label = str(detection.get("label", "objeto"))
        confidence = float(detection.get("confidence", 0.0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        text = f"{label} {confidence:.0%}"
        cv2.putText(
            annotated,
            text,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    api_text = "aguardando inferência"
    if inference is not None:
        api_text = f"API {inference.api_ms:.0f} ms"
        if inference.inference_ms is not None:
            api_text += f" | YOLO {inference.inference_ms:.0f} ms"
    age_text = ""
    if inference_age_s is not None:
        age_text = f" | resultado {inference_age_s:.1f}s"
    osd = f"FPS {stream_fps:.1f} | {api_text}{age_text}"
    cv2.rectangle(annotated, (8, 8), (min(630, 18 + len(osd) * 10), 38), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        osd,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def summarize(records: list[FrameMetrics]) -> dict[str, Any]:
    """Calcula os indicadores exigidos no relatório de diagnóstico."""
    if not records:
        raise ValueError("Não há frames para resumir.")

    totals = [record.total_ms for record in records]
    api_values = [record.api_ms for record in records]
    inference_values = [
        record.inference_ms for record in records if record.inference_ms is not None
    ]
    sorted_totals = sorted(totals)
    p95_index = min(len(sorted_totals) - 1, round((len(sorted_totals) - 1) * 0.95))
    return {
        "frames": len(records),
        "fps_medio": round(1000 / statistics.fmean(totals), 3),
        "latencia_total_media_ms": round(statistics.fmean(totals), 2),
        "latencia_total_p95_ms": round(sorted_totals[p95_index], 2),
        "latencia_api_media_ms": round(statistics.fmean(api_values), 2),
        "latencia_yolo_media_ms": (
            round(statistics.fmean(inference_values), 2) if inference_values else None
        ),
        "frames_com_erro": sum(record.error is not None for record in records),
        "deteccoes_totais": sum(record.detections for record in records),
    }


def write_diagnostic_report(
    output_dir: Path, name: str, records: list[FrameMetrics]
) -> tuple[Path, Path]:
    """Grava dados brutos JSON e um relatório Markdown pronto para entrega."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    json_path = output_dir / f"{name}_diagnostic.json"
    markdown_path = output_dir / f"{name}_report.md"
    json_path.write_text(
        json.dumps(
            {"summary": summary, "frames": [record.as_dict() for record in records]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(
        "\n".join(
            [
                f"# Diagnóstico — {name}",
                "",
                f"Frames analisados: {summary['frames']}",
                f"FPS médio: {summary['fps_medio']}",
                f"Latência total média: {summary['latencia_total_media_ms']} ms",
                f"Latência total p95: {summary['latencia_total_p95_ms']} ms",
                f"Latência média da API: {summary['latencia_api_media_ms']} ms",
                f"Latência média do YOLO: {summary['latencia_yolo_media_ms']} ms",
                f"Detecções totais: {summary['deteccoes_totais']}",
                f"Frames com erro: {summary['frames_com_erro']}",
                "",
                f"Dados por frame: `{json_path.name}`.",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, markdown_path


class LatestInferenceWorker:
    """Executa apenas a inferência mais recente para não acumular atraso."""

    def __init__(self, client: ApiInferenceClient) -> None:
        self.client = client
        self._condition = threading.Condition()
        self._pending: np.ndarray | None = None
        self._latest: tuple[InferenceResult, float] | None = None
        self._stopping = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame_bgr: np.ndarray) -> None:
        with self._condition:
            self._pending = frame_bgr.copy()
            self._condition.notify()

    def latest(self) -> tuple[InferenceResult, float] | None:
        with self._condition:
            return self._latest

    def wait_for_first(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._latest is None and not self._stopping:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return self._latest is not None

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=self.client.config.api_timeout_s + 2)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                frame = self._pending
                self._pending = None
            assert frame is not None
            try:
                result, _ = self.client.infer(frame)
            except InferenceError as exc:
                result = InferenceResult([], 0.0, None, str(exc))
            with self._condition:
                self._latest = (result, time.monotonic())
                self._condition.notify_all()


class LivePipeline:
    """Captura contínua, inferência assíncrona e publicação do último JPEG anotado."""

    def __init__(
        self,
        config: StreamConfig,
        *,
        infer_every: int = 5,
        target_fps: float = 10.0,
    ) -> None:
        if infer_every < 1:
            raise ValueError("infer_every deve ser maior ou igual a 1.")
        self.config = config
        self.infer_every = infer_every
        self.target_fps = target_fps
        self._worker = LatestInferenceWorker(ApiInferenceClient(config))
        self._condition = threading.Condition()
        self._latest_frame: np.ndarray | None = None
        self._latest_jpeg: bytes | None = None
        self._version = 0
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._worker.start()
        self._thread.start()

    def close(self) -> None:
        self._stopping.set()
        self._thread.join(timeout=5)
        self._worker.close()

    def wait_for_frame(
        self, version: int, timeout_s: float = 5.0
    ) -> tuple[int, np.ndarray | None]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._version <= version and not self._stopping.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return version, None
                self._condition.wait(remaining)
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            return self._version, frame

    def wait_for_jpeg(
        self, version: int, timeout_s: float = 5.0
    ) -> tuple[int, bytes | None]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._version <= version and not self._stopping.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return version, None
                self._condition.wait(remaining)
            return self._version, self._latest_jpeg

    def wait_for_inference(self, timeout_s: float) -> bool:
        return self._worker.wait_for_first(timeout_s)

    def _run(self) -> None:
        frame_number = 0
        previous = time.perf_counter()
        interval = 1 / self.target_fps
        with Camera(self.config.width, self.config.height) as camera:
            while not self._stopping.is_set():
                started = time.perf_counter()
                frame, _ = camera.capture()
                frame_number += 1
                if frame_number % self.infer_every == 1:
                    self._worker.submit(frame)
                now = time.perf_counter()
                elapsed = max(now - previous, 0.0001)
                previous = now
                latest = self._worker.latest()
                inference = latest[0] if latest else None
                result_time = latest[1] if latest else None
                annotated = annotate_frame(
                    frame,
                    inference.detections if inference else [],
                    stream_fps=1 / elapsed,
                    inference=inference,
                    inference_age_s=(now - result_time) if result_time else None,
                )
                encoded, jpeg = cv2.imencode(
                    ".jpg",
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
                )
                if encoded:
                    with self._condition:
                        self._latest_frame = annotated
                        self._latest_jpeg = jpeg.tobytes()
                        self._version += 1
                        self._condition.notify_all()
                remaining = interval - (time.perf_counter() - started)
                if remaining > 0:
                    self._stopping.wait(remaining)
