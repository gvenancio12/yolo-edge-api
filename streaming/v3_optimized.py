"""Versão 3: captura e inferência desacopladas, com gravação AVI anotada."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from streaming.pipeline import DEFAULT_EVIDENCE_DIR, LivePipeline, StreamConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100, help="Frames a gravar no AVI.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_EVIDENCE_DIR / "v3_optimized.avi"
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--infer-every", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 1 or args.fps <= 0:
        raise SystemExit("--frames e --fps devem ser positivos.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = StreamConfig(
        api_url=args.api_url,
        confidence=args.confidence,
        width=args.width,
        height=args.height,
    )
    pipeline = LivePipeline(
        config, infer_every=args.infer_every, target_fps=args.fps
    )
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"MJPG"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Não foi possível abrir o AVI em {args.output}.")

    pipeline.start()
    version = 0
    try:
        if not pipeline.wait_for_inference(config.api_timeout_s + 5):
            print("Aviso: a primeira inferência não retornou dentro do tempo esperado.")
        for frame_number in range(1, args.frames + 1):
            version, frame = pipeline.wait_for_frame(version, timeout_s=10)
            if frame is None:
                raise RuntimeError("O pipeline não produziu um frame a tempo.")
            writer.write(frame)
            print(f"Frame gravado: {frame_number}/{args.frames}", flush=True)
    finally:
        writer.release()
        pipeline.close()

    print(f"AVI com bounding boxes e OSD: {args.output}")


if __name__ == "__main__":
    main()
