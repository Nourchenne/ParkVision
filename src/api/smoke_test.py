"""Explicit read-only smoke test of the registered champion on one local image."""

import argparse
import json
import time
from pathlib import Path

from src.api.config import get_settings
from src.api.model_service import load_champion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True,
                        help="One local image from the validation split")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error("validation image does not exist")
    service = load_champion(get_settings())
    start = time.perf_counter()
    result = service.predict(args.image)
    print(json.dumps({"model_version": service.info.version,
                      "architecture": service.info.architecture,
                      "counts": result.counts.model_dump(),
                      "inference_time_ms": round((time.perf_counter() - start) * 1000, 3)},
                     indent=2))


if __name__ == "__main__":
    main()
