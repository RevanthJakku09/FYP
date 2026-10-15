"""
haar_evaluation.py
------------------
Evaluation utilities dedicated to the Haar Cascade pipeline.

Two levels are reported:

1. Detection/localization metrics:
   TP, FP, FN using IoU matching between predicted and ground-truth face boxes.
   These are the appropriate object-detection metrics.

2. Image-level confusion matrix:
   TP, TN, FP, FN for "does this image contain a correctly detected face?"
   This is where TN, Accuracy, Specificity, etc. are defined.

IMPORTANT:
- A true-negative detection box does not exist in ordinary object detection.
  TN is therefore defined at the image level.
- Accuracy/Specificity require negative (no-face) images. The WIDER subset
  currently used by this project contains faces in every image, so without a
  separate negative-image folder those metrics are undefined.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

Box = Tuple[int, int, int, int]


def box_to_corners(box: Box):
    x, y, w, h = box
    return x, y, x + w, y + h


def iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_to_corners(box_a)
    bx1, by1, bx2, by2 = box_to_corners(box_b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_detections(
    predicted_boxes: Sequence[Box],
    ground_truth_boxes: Sequence[Box],
    iou_threshold: float = 0.5,
):
    """Greedy one-to-one matching, highest IoU first."""
    candidates = []
    for pi, pred in enumerate(predicted_boxes):
        for gi, gt in enumerate(ground_truth_boxes):
            score = iou(pred, gt)
            if score >= iou_threshold:
                candidates.append((score, pi, gi))

    candidates.sort(reverse=True, key=lambda item: item[0])

    used_pred = set()
    used_gt = set()
    matches = []

    for score, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, score))

    tp = len(matches)
    fp = len(predicted_boxes) - tp
    fn = len(ground_truth_boxes) - tp

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matches": matches,
    }


def detection_metrics(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def image_confusion(
    predictions: Dict[str, Sequence[Box]],
    ground_truth: Dict[str, Sequence[Box]],
    iou_threshold: float = 0.5,
):
    """
    Build an image-level confusion matrix.

    Positive image = ground truth contains >=1 face.
    Predicted positive = at least one prediction matches a GT face at the
    chosen IoU threshold.

    Negative images must be included in ground_truth with an empty list.
    """
    tp = tn = fp = fn = 0
    per_image = {}

    for name, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(name, [])
        matched = match_detections(pred_boxes, gt_boxes, iou_threshold)
        gt_positive = len(gt_boxes) > 0
        detected_positive = matched["tp"] > 0

        if gt_positive and detected_positive:
            tp += 1
            label = "TP"
        elif gt_positive and not detected_positive:
            fn += 1
            label = "FN"
        elif not gt_positive and pred_boxes:
            fp += 1
            label = "FP"
        else:
            tn += 1
            label = "TN"

        per_image[name] = {
            "classification": label,
            "ground_truth_faces": len(gt_boxes),
            "predicted_faces": len(pred_boxes),
            "matched_faces": matched["tp"],
        }

    return tp, tn, fp, fn, per_image


def classification_metrics(tp: int, tn: int, fp: int, fn: int):
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )

    return {
        "image_true_positives": tp,
        "image_true_negatives": tn,
        "image_false_positives": fp,
        "image_false_negatives": fn,
        "image_accuracy": accuracy,
        "image_precision": precision,
        "image_recall": recall,
        "image_specificity": specificity,
        "image_f1_score": f1,
        "image_count": total,
    }


def evaluate_haar(
    predictions: Dict[str, Sequence[Box]],
    ground_truth: Dict[str, Sequence[Box]],
    iou_threshold: float = 0.5,
):
    """Return both detection-level and image-level metrics."""
    total_tp = total_fp = total_fn = 0
    total_iou = 0.0
    matched_count = 0
    per_image_detection = {}

    for name, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(name, [])
        result = match_detections(pred_boxes, gt_boxes, iou_threshold)

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

        for _, _, score in result["matches"]:
            total_iou += score
            matched_count += 1

        per_image_detection[name] = result

    det = detection_metrics(total_tp, total_fp, total_fn)
    img_tp, img_tn, img_fp, img_fn, per_image_classification = image_confusion(
        predictions, ground_truth, iou_threshold
    )
    cls = classification_metrics(img_tp, img_tn, img_fp, img_fn)

    result = {
        **det,
        **cls,
        "mean_matched_iou": total_iou / matched_count if matched_count else 0.0,
        "iou_threshold": iou_threshold,
        "per_image_detection": per_image_detection,
        "per_image_classification": per_image_classification,
    }
    return result
