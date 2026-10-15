"""
run_haar.py
----------
Standalone Haar Cascade evaluation pipeline.

This file is intentionally separate from the YuNet runners. Existing YuNet
files are not imported or modified.

Examples
--------
# WIDER positive subset only:
python run_haar.py

# Include a folder containing no-face images so TN/Accuracy/Specificity
# become meaningful:
python run_haar.py --negative-dataset datasets/negative

# Save annotated detections:
python run_haar.py --save-images
"""

import argparse
import json
import os
import time

import cv2

from face_detection import (
    load_all_cascades,
    detect_faces_high_recall,
    draw_bounding_boxes,
)
from preprocessing import resize_image
from wider_annotations import load_wider_annotations
from haar_evaluation import evaluate_haar


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


def parse_args():
    p = argparse.ArgumentParser(description="Standalone Haar Cascade evaluation")
    p.add_argument("--dataset", default="datasets/WIDER_subset/images")
    p.add_argument(
        "--annotations",
        default="datasets/WIDER_subset/annotations/wider_face_val_bbx_gt.txt",
    )
    p.add_argument(
        "--negative-dataset",
        default=None,
        help="Folder of images containing no faces. Needed for TN/accuracy/specificity.",
    )
    p.add_argument("--output", default="output/haar")
    p.add_argument("--scale-factor", type=float, default=1.05)
    p.add_argument("--min-neighbors", type=int, default=4)
    p.add_argument("--min-size", type=int, nargs=2, default=(10, 10))
    p.add_argument("--min-votes", type=int, default=2)
    p.add_argument("--merge-iou-threshold", type=float, default=0.40)
    p.add_argument("--max-dimension", type=int, default=1200)
    p.add_argument("--upscale-factor", type=float, default=1.5)
    p.add_argument("--no-upscale", action="store_true")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--save-images", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("\n" + "=" * 72)
    print("HAAR CASCADE FACE DETECTOR — STANDALONE EVALUATION")
    print("=" * 72)
    print("YuNet files: NOT MODIFIED / NOT USED")
    print(f"IoU threshold       : {args.iou_threshold}")
    print(f"scaleFactor         : {args.scale_factor}")
    print(f"minNeighbors        : {args.min_neighbors}")
    print(f"minSize             : {tuple(args.min_size)}")
    print(f"minVotes             : {args.min_votes}")
    print(f"upscale              : {not args.no_upscale}")

    gt_original = load_wider_annotations(args.annotations)
    image_paths = find_images(args.dataset)
    negative_paths = find_images(args.negative_dataset)

    cascades = load_all_cascades()
    predictions = {}
    ground_truth = {}
    total_detection_time = 0.0
    processed = 0

    # Positive/annotated images
    for index, image_path in enumerate(image_paths, 1):
        key = key_for(image_path, args.dataset)
        image = cv2.imread(image_path)
        if image is None or key not in gt_original:
            continue

        original_h, original_w = image.shape[:2]
        resized = resize_image(image, max_dimension=args.max_dimension)
        resized_h, resized_w = resized.shape[:2]

        sx = resized_w / original_w
        sy = resized_h / original_h
        gt_boxes = scale_boxes(gt_original[key], sx, sy)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        start = time.perf_counter()
        boxes = detect_faces_high_recall(
            gray,
            cascades,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=tuple(args.min_size),
            use_upscale=not args.no_upscale,
            upscale_factor=args.upscale_factor,
            merge_iou_threshold=args.merge_iou_threshold,
            min_votes=args.min_votes,
        )
        total_detection_time += time.perf_counter() - start
        processed += 1

        predictions[key] = boxes
        ground_truth[key] = gt_boxes

        if args.save_images:
            destination = os.path.join(args.output, key)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            cv2.imwrite(destination, draw_bounding_boxes(resized, boxes))

        print(
            f"{index:3d}/{len(image_paths):3d} {key} -> "
            f"{len(boxes)} detections"
        )

    # Negative images are explicitly represented by an empty GT list.
    for index, image_path in enumerate(negative_paths, 1):
        key = "NEGATIVE/" + key_for(image_path, args.negative_dataset)
        image = cv2.imread(image_path)
        if image is None:
            continue

        resized = resize_image(image, max_dimension=args.max_dimension)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        start = time.perf_counter()
        boxes = detect_faces_high_recall(
            gray,
            cascades,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=tuple(args.min_size),
            use_upscale=not args.no_upscale,
            upscale_factor=args.upscale_factor,
            merge_iou_threshold=args.merge_iou_threshold,
            min_votes=args.min_votes,
        )
        total_detection_time += time.perf_counter() - start
        processed += 1

        predictions[key] = boxes
        ground_truth[key] = []

        if args.save_images:
            destination = os.path.join(args.output, key)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            cv2.imwrite(destination, draw_bounding_boxes(resized, boxes))

        print(
            f"NEG {index:3d}/{len(negative_paths):3d} {key} -> "
            f"{len(boxes)} detections"
        )

    metrics = evaluate_haar(
        predictions,
        ground_truth,
        iou_threshold=args.iou_threshold,
    )

    metrics["images_processed"] = processed
    metrics["positive_images"] = sum(bool(v) for v in ground_truth.values())
    metrics["negative_images"] = sum(not bool(v) for v in ground_truth.values())
    metrics["total_detection_time_ms"] = total_detection_time * 1000.0
    metrics["mean_detection_latency_ms"] = (
        total_detection_time / processed * 1000.0 if processed else 0.0
    )
    metrics["fps"] = (
        processed / total_detection_time if total_detection_time else 0.0
    )

    # Accuracy/specificity are deliberately N/A when there are no negative images.
    if metrics["negative_images"] == 0:
        metrics["image_accuracy"] = None
        metrics["image_specificity"] = None

    result_path = os.path.join(args.output, "haar_metrics.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    def pct(value):
        return "N/A" if value is None else f"{value:.4f}"

    print("\n" + "=" * 72)
    print("HAAR CASCADE RESULTS")
    print("=" * 72)
    print("Detection-level (object detection)")
    print(f"TP                 : {metrics['true_positives']}")
    print(f"FP                 : {metrics['false_positives']}")
    print(f"FN                 : {metrics['false_negatives']}")
    print(f"Precision          : {metrics['precision']:.4f}")
    print(f"Recall             : {metrics['recall']:.4f}")
    print(f"F1 Score           : {metrics['f1_score']:.4f}")
    print(f"Mean matched IoU   : {metrics['mean_matched_iou']:.4f}")
    print()
    print("Image-level classification")
    print(f"TP (images)        : {metrics['image_true_positives']}")
    print(f"TN (images)        : {metrics['image_true_negatives']}")
    print(f"FP (images)        : {metrics['image_false_positives']}")
    print(f"FN (images)        : {metrics['image_false_negatives']}")
    print(f"Accuracy           : {pct(metrics['image_accuracy'])}")
    print(f"Specificity        : {pct(metrics['image_specificity'])}")
    print(f"Precision          : {pct(metrics['image_precision'])}")
    print(f"Recall             : {pct(metrics['image_recall'])}")
    print(f"F1 Score           : {pct(metrics['image_f1_score'])}")
    print()
    print("Performance")
    print(f"Images processed   : {processed}")
    print(f"Mean latency       : {metrics['mean_detection_latency_ms']:.2f} ms")
    print(f"Throughput         : {metrics['fps']:.2f} FPS")
    print(f"Results JSON       : {result_path}")

    if not negative_paths:
        print()
        print("NOTE: No negative-image folder was supplied.")
        print("      Current WIDER subset contains face-positive images only.")
        print("      Add no-face images under datasets/negative and rerun with:")
        print("      python run_haar.py --negative-dataset datasets/negative")


if __name__ == "__main__":
    main()
