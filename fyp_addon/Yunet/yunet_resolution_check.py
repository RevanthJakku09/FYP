import argparse
import json
import os
import time

import cv2

from haar_evaluation import evaluate_haar
from preprocessing import resize_image
from wider_annotations import load_wider_annotations
from yunet_detector import (
    load_yunet,
    detect_faces_yunet,
    draw_yunet_boxes,
)

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def find_images(root):
    paths = []

    if not root or not os.path.isdir(root):
        return paths

    for directory, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(EXTENSIONS):
                paths.append(os.path.join(directory, name))

    return sorted(paths)


def key_for(path, dataset):
    return os.path.relpath(path, dataset).replace(os.sep, "/")


def scale_boxes(boxes, sx, sy):
    return [
        (
            int(round(x * sx)),
            int(round(y * sy)),
            int(round(w * sx)),
            int(round(h * sy)),
        )
        for x, y, w, h in boxes
    ]


def parse_size(size_string):
    """Convert 640x480 into (640, 480)."""

    try:
        width, height = size_string.lower().split("x")
        width = int(width)
        height = int(height)

        if width <= 0 or height <= 0:
            raise ValueError

        return width, height

    except ValueError:
        raise argparse.ArgumentTypeError(
            "Size must be written as WIDTHxHEIGHT, e.g. 640x480"
        )


def resize_fixed_letterbox(image, out_width, out_height):
    """
    Resize while preserving aspect ratio and add black padding.

    Returns:
        output_image, transform
    """

    original_h, original_w = image.shape[:2]

    scale = min(
        out_width / original_w,
        out_height / original_h,
    )

    new_width = max(
        1,
        int(round(original_w * scale))
    )

    new_height = max(
        1,
        int(round(original_h * scale))
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = cv2.copyMakeBorder(
        resized,
        0,
        out_height - new_height,
        0,
        out_width - new_width,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    # Center the image inside the output.
    pad_x = (out_width - new_width) // 2
    pad_y = (out_height - new_height) // 2

    canvas = cv2.copyMakeBorder(
        resized,
        pad_y,
        out_height - new_height - pad_y,
        pad_x,
        out_width - new_width - pad_x,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    transform = (
        new_width / original_w,
        new_height / original_h,
        pad_x,
        pad_y,
    )

    return canvas, transform


def parse_args():

    parser = argparse.ArgumentParser(
        description="YuNet fixed-size preprocessing evaluation"
    )

    parser.add_argument(
        "--dataset",
        default="datasets/WIDER_subset/images"
    )

    parser.add_argument(
        "--annotations",
        default=(
            "datasets/WIDER_subset/annotations/"
            "wider_face_val_bbx_gt.txt"
        )
    )

    parser.add_argument(
        "--negative-dataset",
        default="datasets/negative"
    )

    parser.add_argument(
        "--model",
        default="models/face_detection_yunet_2023mar.onnx"
    )

    parser.add_argument(
        "--output",
        default="output/yunet_size_test"
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.6
    )

    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.3
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--size",
        type=parse_size,
        required=True,
        help="Fixed output size, e.g. 320x240"
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--save-images",
        action="store_true"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    output_width, output_height = args.size

    os.makedirs(
        args.output,
        exist_ok=True
    )

    print("\n" + "=" * 72)
    print("YUNET FIXED-SIZE EVALUATION")
    print("=" * 72)

    print(
        f"Model              : {args.model}"
    )

    print(
        f"Output size        : "
        f"{output_width} x {output_height}"
    )

    print(
        f"Score threshold    : {args.score_threshold}"
    )

    print(
        f"NMS threshold      : {args.nms_threshold}"
    )

    print(
        f"IoU threshold      : {args.iou_threshold}"
    )

    # ---------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------

    gt_original = load_wider_annotations(
        args.annotations
    )

    positive_paths = find_images(
        args.dataset
    )

    negative_paths = find_images(
        args.negative_dataset
    )

    print(
        f"\nPositive images    : "
        f"{len(positive_paths)}"
    )

    print(
        f"Negative images    : "
        f"{len(negative_paths)}"
    )

    print(
        f"Annotated images   : "
        f"{len(gt_original)}"
    )

    # ---------------------------------------------------------
    # Load YuNet
    # ---------------------------------------------------------

    detector = load_yunet(
        args.model,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        top_k=args.top_k,
    )

    predictions = {}
    ground_truth = {}

    total_preprocessing_time = 0.0
    total_detection_time = 0.0

    # =========================================================
    # POSITIVE DATASET
    # =========================================================

    print("\n" + "-" * 72)
    print("PROCESSING POSITIVE WIDER IMAGES")
    print("-" * 72)

    for index, image_path in enumerate(
        positive_paths,
        1
    ):

        key = key_for(
            image_path,
            args.dataset
        )

        image = cv2.imread(
            image_path
        )

        if image is None:
            continue

        if key not in gt_original:
            continue

        original_h, original_w = image.shape[:2]

        # -----------------------------------------------------
        # FIXED-SIZE RESIZE
        # -----------------------------------------------------

        preprocessing_start = time.perf_counter()

        resized, transform = resize_fixed_letterbox(
            image,
            output_width,
            output_height,
        )

        preprocessing_elapsed = (
            time.perf_counter()
            - preprocessing_start
        )

        total_preprocessing_time += (
            preprocessing_elapsed
        )

        # -----------------------------------------------------
        # Transform ground-truth boxes
        # -----------------------------------------------------

        sx, sy, pad_x, pad_y = transform

        ground_truth[key] = [
            (
                int(round(x * sx + pad_x)),
                int(round(y * sy + pad_y)),
                int(round(w * sx)),
                int(round(h * sy)),
            )
            for x, y, w, h in gt_original[key]
        ]

        # -----------------------------------------------------
        # YUNET
        # -----------------------------------------------------

        detection_start = time.perf_counter()

        boxes = detect_faces_yunet(
            resized,
            detector
        )

        detection_elapsed = (
            time.perf_counter()
            - detection_start
        )

        total_detection_time += (
            detection_elapsed
        )

        predictions[key] = boxes

        print(
            f"{index:3d}/{len(positive_paths):3d} "
            f"{key} -> {len(boxes)} detections "
            f"(Pre: {preprocessing_elapsed * 1000:.2f} ms, "
            f"YuNet: {detection_elapsed * 1000:.2f} ms)"
        )

        if args.save_images:

            output_image = draw_yunet_boxes(
                resized,
                boxes
            )

            destination = os.path.join(
                args.output,
                key
            )

            os.makedirs(
                os.path.dirname(destination),
                exist_ok=True
            )

            cv2.imwrite(
                destination,
                output_image
            )

    # =========================================================
    # NEGATIVE DATASET
    # =========================================================

    print("\n" + "-" * 72)
    print("PROCESSING NEGATIVE IMAGES")
    print("-" * 72)

    for index, image_path in enumerate(
        negative_paths,
        1
    ):

        negative_key = (
            "NEGATIVE/"
            + key_for(
                image_path,
                args.negative_dataset
            )
        )

        image = cv2.imread(
            image_path
        )

        if image is None:
            continue

        preprocessing_start = time.perf_counter()

        resized, _ = resize_fixed_letterbox(
            image,
            output_width,
            output_height,
        )

        preprocessing_elapsed = (
            time.perf_counter()
            - preprocessing_start
        )

        total_preprocessing_time += (
            preprocessing_elapsed
        )

        ground_truth[negative_key] = []

        detection_start = time.perf_counter()

        boxes = detect_faces_yunet(
            resized,
            detector
        )

        detection_elapsed = (
            time.perf_counter()
            - detection_start
        )

        total_detection_time += (
            detection_elapsed
        )

        predictions[negative_key] = boxes

        print(
            f"NEG {index:3d}/"
            f"{len(negative_paths):3d} "
            f"{negative_key} -> "
            f"{len(boxes)} detections "
            f"(Pre: {preprocessing_elapsed * 1000:.2f} ms, "
            f"YuNet: {detection_elapsed * 1000:.2f} ms)"
        )

        if args.save_images:

            output_image = draw_yunet_boxes(
                resized,
                boxes
            )

            destination = os.path.join(
                args.output,
                negative_key
            )

            os.makedirs(
                os.path.dirname(destination),
                exist_ok=True
            )

            cv2.imwrite(
                destination,
                output_image
            )

    # =========================================================
    # EVALUATION
    # =========================================================

    metrics = evaluate_haar(
        predictions,
        ground_truth,
        iou_threshold=args.iou_threshold
    )

    count = len(predictions)

    mean_detection_latency_ms = (
        total_detection_time / count * 1000
        if count
        else 0.0
    )

    mean_preprocessing_latency_ms = (
        total_preprocessing_time / count * 1000
        if count
        else 0.0
    )

    total_processing_time = (
        total_preprocessing_time
        + total_detection_time
    )

    mean_total_latency_ms = (
        total_processing_time / count * 1000
        if count
        else 0.0
    )

    fps = (
        count / total_processing_time
        if total_processing_time
        else 0.0
    )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    metrics["input_output_size"] = (
        f"{output_width}x{output_height}"
    )

    metrics["images_processed"] = count

    metrics["total_preprocessing_time_ms"] = (
        total_preprocessing_time * 1000
    )

    metrics["mean_preprocessing_latency_ms"] = (
        mean_preprocessing_latency_ms
    )

    metrics["total_detection_time_ms"] = (
        total_detection_time * 1000
    )

    metrics["mean_detection_latency_ms"] = (
        mean_detection_latency_ms
    )

    metrics["total_processing_time_ms"] = (
        total_processing_time * 1000
    )

    metrics["mean_total_processing_latency_ms"] = (
        mean_total_latency_ms
    )

    metrics["fps"] = fps

    result_path = os.path.join(
        args.output,
        f"yunet_{output_width}x{output_height}_metrics.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # =========================================================
    # RESULTS
    # =========================================================

    print("\n" + "=" * 72)
    print("YUNET FIXED-SIZE RESULTS")
    print("=" * 72)

    print(
        f"Output size        : "
        f"{output_width} x {output_height}"
    )

    print(
        f"Precision          : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall             : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score           : "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"Mean matched IoU   : "
        f"{metrics['mean_matched_iou']:.4f}"
    )

    print("\nPerformance")

    print(
        f"Mean preprocessing : "
        f"{mean_preprocessing_latency_ms:.2f} ms/image"
    )

    print(
        f"Mean YuNet time    : "
        f"{mean_detection_latency_ms:.2f} ms/image"
    )

    print(
        f"Mean total latency : "
        f"{mean_total_latency_ms:.2f} ms/image"
    )

    print(
        f"Throughput         : "
        f"{fps:.2f} FPS"
    )

    print(
        f"Results JSON       : "
        f"{result_path}"
    )


if __name__ == "__main__":
    main()