"""Versão 1: captura síncrona e uma chamada HTTP por frame."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from streaming.pipeline import (
    ApiInferenceClient,
    Camera,
    DEFAULT_EVIDENCE_DIR,
    FrameMetrics,
    StreamConfig,
    summarize,
    write_diagnostic_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=50, help="Quantidade de frames.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 50:
        raise SystemExit("Use ao menos 50 frames para cumprir o diagnóstico da aula.")

    config = StreamConfig(
        api_url=args.api_url,
        confidence=args.confidence,
        width=args.width,
        height=args.height,
    )
    client = ApiInferenceClient(config)
    records: list[FrameMetrics] = []

    with Camera(config.width, config.height) as camera:
        for frame_number in range(1, args.frames + 1):
            started = time.perf_counter()
            frame, capture_ms = camera.capture()
            result, encode_ms = client.infer(frame)
            total_ms = (time.perf_counter() - started) * 1000
            record = FrameMetrics(
                frame=frame_number,
                capture_ms=round(capture_ms, 2),
                encode_ms=round(encode_ms, 2),
                api_ms=round(result.api_ms, 2),
                inference_ms=result.inference_ms,
                total_ms=round(total_ms, 2),
                detections=len(result.detections),
                error=result.error,
            )
            records.append(record)
            print(record.as_dict(), flush=True)

    json_path, markdown_path = write_diagnostic_report(
        args.output_dir, "v1_naive", records
    )
    print("Resumo:", summarize(records))
    print(f"Dados brutos: {json_path}")
    print(f"Relatório: {markdown_path}")


if __name__ == "__main__":
    main()
