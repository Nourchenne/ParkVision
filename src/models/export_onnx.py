"""Export a trained ParkVision YOLO checkpoint to ONNX and benchmark inference latency.

Usage:
    python -m src.models.export_onnx --checkpoint path/to/best.pt --image path/to/sample.jpg
"""

import argparse
import json
import time
from pathlib import Path


def export_to_onnx(checkpoint: Path, output: Path | None = None, yolo_factory=None) -> Path:
    """Export a YOLO .pt checkpoint to ONNX; returns the produced .onnx path."""
    if yolo_factory is None:
        from ultralytics import YOLO
        yolo_factory = YOLO
    model = yolo_factory(str(checkpoint))
    exported = Path(model.export(format="onnx", imgsz=640, dynamic=False))
    if output is not None and exported != output:
        output.parent.mkdir(parents=True, exist_ok=True)
        exported.replace(output)
        exported = output
    return exported


def benchmark_pytorch(checkpoint: Path, image: Path, runs: int = 20, yolo_factory=None) -> float:
    """Average inference latency in ms using the ultralytics PyTorch runtime."""
    if yolo_factory is None:
        from ultralytics import YOLO
        yolo_factory = YOLO
    model = yolo_factory(str(checkpoint))
    model.predict(str(image), verbose=False)  # warm-up, excluded from the measurement
    started = time.perf_counter()
    for _ in range(runs):
        model.predict(str(image), verbose=False)
    return (time.perf_counter() - started) * 1000 / runs


def _preprocess(image: Path, imgsz: int):
    import numpy as np
    from PIL import Image

    with Image.open(image) as source:
        frame = source.convert("RGB").resize((imgsz, imgsz))
    array = np.asarray(frame, dtype="float32") / 255.0
    return array.transpose(2, 0, 1)[None, ...]


def benchmark_onnxruntime(onnx_path: Path, image: Path, imgsz: int = 640, runs: int = 20,
                          session_factory=None) -> float:
    """Average inference latency in ms using ONNX Runtime directly (no ultralytics overhead)."""
    if session_factory is None:
        import onnxruntime as ort

        def session_factory(path):
            return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    session = session_factory(str(onnx_path))
    input_name = session.get_inputs()[0].name
    tensor = _preprocess(image, imgsz)
    session.run(None, {input_name: tensor})  # warm-up
    started = time.perf_counter()
    for _ in range(runs):
        session.run(None, {input_name: tensor})
    return (time.perf_counter() - started) * 1000 / runs


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Local YOLO .pt checkpoint to export")
    parser.add_argument("--image", required=True, type=Path,
                        help="Sample image used for the latency benchmark")
    parser.add_argument("--output", type=Path, default=None,
                        help="Destination .onnx path (default: next to the checkpoint)")
    parser.add_argument("--runs", type=int, default=20,
                        help="Number of timed inference passes per runtime")
    args = parser.parse_args(argv)

    onnx_path = export_to_onnx(args.checkpoint, args.output)
    pytorch_ms = benchmark_pytorch(args.checkpoint, args.image, args.runs)
    onnx_ms = benchmark_onnxruntime(onnx_path, args.image, runs=args.runs)
    report = {
        "onnx_path": str(onnx_path),
        "runs": args.runs,
        "pytorch_inference_ms": round(pytorch_ms, 3),
        "onnxruntime_inference_ms": round(onnx_ms, 3),
        "speedup": round(pytorch_ms / onnx_ms, 3) if onnx_ms else None,
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()