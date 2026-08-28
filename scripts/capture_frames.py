"""Captura frames EPI da câmera CSI e descarta imagens muito parecidas.

Execute na Raspberry Pi:
    python3 scripts/capture_frames.py --count 180
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_OUTPUT_DIR = Path.home() / "epi-capture" / "raw"


@dataclass
class CaptureStats:
    saved: int = 0
    discarded_duplicate: int = 0
    discarded_error: int = 0
    attempts: int = 0


def frame_change_score(previous: np.ndarray, current: np.ndarray) -> float:
    """Retorna a diferença média entre dois frames em escala de cinza."""
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(previous_gray, current_gray).mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=180, help="Mínimo de frames salvos.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--min-change",
        type=float,
        default=2.0,
        help="Diferença média mínima para salvar um novo frame; use 0 para não filtrar.",
    )
    parser.add_argument("--interval", type=float, default=0.35)
    parser.add_argument("--max-attempts", type=int, default=1800)
    return parser.parse_args()


def open_camera(width: int, height: int) -> Any:
    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise RuntimeError("Picamera2 só está disponível na Raspberry Pi.") from exc

    camera = Picamera2()
    config = camera.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    time.sleep(0.8)
    return camera


def main() -> None:
    args = parse_args()
    if args.count < 150:
        raise SystemExit("Use --count >= 150 para cumprir a atividade.")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width e --height devem ser positivos.")
    if args.interval < 0 or args.min_change < 0:
        raise SystemExit("--interval e --min-change não podem ser negativos.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = CaptureStats()
    previous: np.ndarray | None = None
    started = time.perf_counter()
    camera = open_camera(args.width, args.height)

    print(f"Capturando {args.count} frames em {args.output_dir}", flush=True)
    print(
        f"Filtro de duplicados: diferença mínima = {args.min_change:.2f}",
        flush=True,
    )
    try:
        while stats.saved < args.count and stats.attempts < args.max_attempts:
            stats.attempts += 1
            try:
                raw_frame = camera.capture_array("main")
                frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                if previous is not None:
                    change = frame_change_score(previous, frame)
                    if change < args.min_change:
                        stats.discarded_duplicate += 1
                        time.sleep(args.interval)
                        continue

                filename = args.output_dir / f"epi_{stats.saved + 1:04d}.jpg"
                saved = cv2.imwrite(
                    str(filename),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
                )
                if not saved:
                    stats.discarded_error += 1
                    continue
                previous = frame
                stats.saved += 1
                if stats.saved % 10 == 0 or stats.saved == args.count:
                    print(
                        "progresso "
                        f"salvos={stats.saved}/{args.count} "
                        f"descartados={stats.discarded_duplicate} "
                        f"tentativas={stats.attempts}",
                        flush=True,
                    )
            except Exception as exc:  # câmera deve continuar tentando em falha transitória
                stats.discarded_error += 1
                print(f"aviso tentativa={stats.attempts}: {exc}", flush=True)
            time.sleep(args.interval)
    finally:
        camera.stop()
        camera.close()

    elapsed_s = round(time.perf_counter() - started, 2)
    summary = {
        **asdict(stats),
        "elapsed_s": elapsed_s,
        "output_dir": str(args.output_dir),
        "target": args.count,
        "completed": stats.saved >= args.count,
    }
    (args.output_dir / "capture_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("CAPTURA CONCLUÍDA" if summary["completed"] else "CAPTURA INCOMPLETA")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not summary["completed"]:
        raise SystemExit("Limite de tentativas atingido antes da meta de frames.")


if __name__ == "__main__":
    main()
