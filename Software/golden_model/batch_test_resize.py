"""
batch_test_resize.py

Batch verification for the FPGA resize Python golden model.

It recursively scans a dataset folder, runs the fixed-point golden model
on every supported image, and writes a CSV report.

Default output resolution: 960x720.
"""

import csv
import sys
import time
from pathlib import Path

from PIL import Image

# Import our golden model from the same folder.
from resize_golden import resize_fixed


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def find_images(dataset_dir):
    """Find supported images recursively."""
    dataset_dir = Path(dataset_dir)

    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(
            r"python batch_test_resize.py "
            r"..\datasets"
        )
        sys.exit(1)

    dataset_dir = Path(sys.argv[1])

    if not dataset_dir.exists():
        print(f"ERROR: Dataset folder not found: {dataset_dir}")
        sys.exit(1)

    if not dataset_dir.is_dir():
        print(f"ERROR: Not a folder: {dataset_dir}")
        sys.exit(1)

    # Keep the report outside the dataset so the dataset is not modified.
    report_path = Path("../output/resize_dataset_report.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    total = 0
    passed = 0
    failed = 0

    images = list(find_images(dataset_dir))

    if not images:
        print(f"No supported images found in: {dataset_dir}")
        sys.exit(1)

    print(f"Dataset : {dataset_dir.resolve()}")
    print(f"Images  : {len(images)}")
    print("Output  : 960x720")
    print()

    for index, image_path in enumerate(images, start=1):
        total += 1

        try:
            with Image.open(image_path) as im:
                image = im.convert("RGB")
                in_w, in_h = image.size

                start = time.perf_counter()

                output, params = resize_fixed(
                    __import__("numpy").array(image),
                    out_w=960,
                    out_h=720
                )

                elapsed_ms = (time.perf_counter() - start) * 1000

            out_h, out_w, channels = output.shape

            dimensions_ok = (out_w == 960 and out_h == 720)
            pixels_ok = (
                output.dtype.name == "uint8"
                and int(output.min()) >= 0
                and int(output.max()) <= 255
            )

            status = "PASS" if dimensions_ok and pixels_ok else "FAIL"

            if status == "PASS":
                passed += 1
            else:
                failed += 1

            rows.append({
                "file": str(image_path.relative_to(dataset_dir)),
                "input_width": in_w,
                "input_height": in_h,
                "scale": params["scale"],
                "scaled_width": params["scaled_w"],
                "scaled_height": params["scaled_h"],
                "pad_left": params["pad_left"],
                "pad_right": params["pad_right"],
                "pad_top": params["pad_top"],
                "pad_bottom": params["pad_bottom"],
                "output_width": out_w,
                "output_height": out_h,
                "pixel_min": int(output.min()),
                "pixel_max": int(output.max()),
                "time_ms": round(elapsed_ms, 3),
                "status": status,
                "error": ""
            })

            print(
                f"[{index:>4}/{len(images)}] "
                f"{status:<4} "
                f"{in_w}x{in_h} -> 960x720  "
                f"{image_path.name}"
            )

        except Exception as exc:
            failed += 1

            rows.append({
                "file": str(image_path.relative_to(dataset_dir)),
                "input_width": "",
                "input_height": "",
                "scale": "",
                "scaled_width": "",
                "scaled_height": "",
                "pad_left": "",
                "pad_right": "",
                "pad_top": "",
                "pad_bottom": "",
                "output_width": "",
                "output_height": "",
                "pixel_min": "",
                "pixel_max": "",
                "time_ms": "",
                "status": "FAIL",
                "error": str(exc)
            })

            print(
                f"[{index:>4}/{len(images)}] "
                f"FAIL {image_path.name}: {exc}"
            )

    fieldnames = [
        "file",
        "input_width",
        "input_height",
        "scale",
        "scaled_width",
        "scaled_height",
        "pad_left",
        "pad_right",
        "pad_top",
        "pad_bottom",
        "output_width",
        "output_height",
        "pixel_min",
        "pixel_max",
        "time_ms",
        "status",
        "error",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print("BATCH TEST COMPLETE")
    print("=" * 60)
    print(f"Total : {total}")
    print(f"Pass  : {passed}")
    print(f"Fail  : {failed}")
    print(f"Report: {report_path.resolve()}")

    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
