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
    return os.path.relpath(
        path, dataset
    ).replace(os.sep, "/")


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


def parse_args():

    parser = argparse.ArgumentParser(
        description="Complete YuNet evaluation"
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
        default="output/yunet_complete"
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
        "--max-dimension",
        type=int,
        default=1200
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

    os.makedirs(
        args.output,
        exist_ok=True
    )

    print("\n" + "=" * 72)
    print("YUNET COMPLETE EVALUATION")
    print("=" * 72)

    print(
        f"Model              : {args.model}"
    )

    print(
        f"Score threshold    : {args.score_threshold}"
    )

    print(
        f"NMS threshold      : {args.nms_threshold}"
    )

    print(
        f"Top-K              : {args.top_k}"
    )

    print(
        f"Max dimension      : {args.max_dimension}"
    )

    print(
        f"IoU threshold      : {args.iou_threshold}"
    )

    print(
        f"Negative dataset   : {args.negative_dataset}"
    )

    # ---------------------------------------------------------
    # Load ground truth
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
        f"\nPositive images    : {len(positive_paths)}"
    )

    print(
        f"Negative images    : {len(negative_paths)}"
    )

    print(
        f"Annotated images   : {len(gt_original)}"
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

    # =========================================================
    # TIMING VARIABLES
    # =========================================================

    total_preprocessing_time = 0.0
    total_detection_time = 0.0

    # =========================================================
    # POSITIVE DATASET — WIDER
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

            print(
                f"{index}/{len(positive_paths)} "
                f"ERROR: {key}"
            )

            continue

        if key not in gt_original:
            continue

        original_h, original_w = image.shape[:2]

        # -----------------------------------------------------
        # PREPROCESSING TIMING
        # -----------------------------------------------------

        preprocessing_start = time.perf_counter()

        resized = resize_image(
            image,
            max_dimension=args.max_dimension
        )

        preprocessing_elapsed = (
            time.perf_counter()
            - preprocessing_start
        )

        total_preprocessing_time += (
            preprocessing_elapsed
        )

        # -----------------------------------------------------
        # Scale ground-truth boxes
        # -----------------------------------------------------

        resized_h, resized_w = resized.shape[:2]

        sx = resized_w / original_w
        sy = resized_h / original_h

        ground_truth[key] = scale_boxes(
            gt_original[key],
            sx,
            sy
        )

        # -----------------------------------------------------
        # YUNET DETECTION TIMING
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

            print(
                f"NEG {index:3d}/"
                f"{len(negative_paths):3d} "
                f"ERROR: {negative_key}"
            )

            continue

        # -----------------------------------------------------
        # PREPROCESSING TIMING
        # -----------------------------------------------------

        preprocessing_start = time.perf_counter()

        resized = resize_image(
            image,
            max_dimension=args.max_dimension
        )

        preprocessing_elapsed = (
            time.perf_counter()
            - preprocessing_start
        )

        total_preprocessing_time += (
            preprocessing_elapsed
        )

        # Negative image has ZERO ground-truth faces.
        ground_truth[negative_key] = []

        # -----------------------------------------------------
        # YUNET DETECTION TIMING
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

    # ---------------------------------------------------------
    # Detection timing
    # ---------------------------------------------------------

    mean_detection_latency_ms = (
        total_detection_time
        / count
        * 1000
        if count
        else 0.0
    )

    # ---------------------------------------------------------
    # Preprocessing timing
    # ---------------------------------------------------------

    mean_preprocessing_latency_ms = (
        total_preprocessing_time
        / count
        * 1000
        if count
        else 0.0
    )

    # ---------------------------------------------------------
    # Total processing timing
    # ---------------------------------------------------------

    total_processing_time = (
        total_preprocessing_time
        + total_detection_time
    )

    mean_total_latency_ms = (
        total_processing_time
        / count
        * 1000
        if count
        else 0.0
    )

    fps = (
        count / total_processing_time
        if total_processing_time
        else 0.0
    )

    # ---------------------------------------------------------
    # Add timing information to metrics
    # ---------------------------------------------------------

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

    metrics["positive_images"] = sum(
        bool(v)
        for v in ground_truth.values()
    )

    metrics["negative_images"] = sum(
        not bool(v)
        for v in ground_truth.values()
    )

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------

    result_path = os.path.join(
        args.output,
        "yunet_complete_metrics.json"
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

    def fmt(value):

        if value is None:
            return "N/A"

        return f"{value:.4f}"

    print("\n" + "=" * 72)
    print("YUNET COMPLETE RESULTS")
    print("=" * 72)

    # ---------------------------------------------------------
    # Detection-level metrics
    # ---------------------------------------------------------

    print(
        "Detection-level (object detection)"
    )

    print(
        f"TP                 : "
        f"{metrics['true_positives']}"
    )

    print(
        f"FP                 : "
        f"{metrics['false_positives']}"
    )

    print(
        f"FN                 : "
        f"{metrics['false_negatives']}"
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

    # ---------------------------------------------------------
    # Image-level metrics
    # ---------------------------------------------------------

    print("\nImage-level classification")

    print(
        f"TP (images)        : "
        f"{metrics['image_true_positives']}"
    )

    print(
        f"TN (images)        : "
        f"{metrics['image_true_negatives']}"
    )

    print(
        f"FP (images)        : "
        f"{metrics['image_false_positives']}"
    )

    print(
        f"FN (images)        : "
        f"{metrics['image_false_negatives']}"
    )

    print(
        f"Accuracy           : "
        f"{fmt(metrics['image_accuracy'])}"
    )

    print(
        f"Specificity        : "
        f"{fmt(metrics['image_specificity'])}"
    )

    print(
        f"Precision          : "
        f"{fmt(metrics['image_precision'])}"
    )

    print(
        f"Recall             : "
        f"{fmt(metrics['image_recall'])}"
    )

    print(
        f"F1 Score           : "
        f"{fmt(metrics['image_f1_score'])}"
    )

    # ---------------------------------------------------------
    # Performance
    # ---------------------------------------------------------

    print("\nPerformance")

    print(
        f"Images processed          : "
        f"{count}"
    )

    print(
        f"Total preprocessing time  : "
        f"{total_preprocessing_time * 1000:.2f} ms"
    )

    print(
        f"Mean preprocessing time   : "
        f"{mean_preprocessing_latency_ms:.2f} ms/image"
    )

    print(
        f"Total YuNet detection time: "
        f"{total_detection_time * 1000:.2f} ms"
    )

    print(
        f"Mean YuNet detection time : "
        f"{mean_detection_latency_ms:.2f} ms/image"
    )

    print(
        f"Total processing time     : "
        f"{total_processing_time * 1000:.2f} ms"
    )

    print(
        f"Mean total latency        : "
        f"{mean_total_latency_ms:.2f} ms/image"
    )

    print(
        f"Throughput                : "
        f"{fps:.2f} FPS"
    )

    print(
        f"Results JSON              : "
        f"{result_path}"
    )


if __name__ == "__main__":
    main()