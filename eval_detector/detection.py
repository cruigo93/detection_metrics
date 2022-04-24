import numpy as np
from collections import Counter
from loguru import logger


def intersection_over_union(predicted_bboxes, target_bboxes):
    """
    Calculates rowise bounding box IoUs between two bounding box tensors

    Parameters:
        predicted_bboxes: Numpy Array containing a batch of predicted bounding boxes
            type:tensor
            shape:[N,4]
            format:[x1,y1,x2,y2]
        target_bboxes: Numpy Array containing a batch of ground truth bounding boxes
            type:tensor
            shape:[N,4]
            format:[x1,y1,x2,y2]
    Result:
        IoUs: Batch of IoUs
        type: tensor
        shape: 1D tensor of size N

    """
    box1_x1 = predicted_bboxes[:, 0]
    box1_y1 = predicted_bboxes[:, 1]
    box1_x2 = predicted_bboxes[:, 2]
    box1_y2 = predicted_bboxes[:, 3]
    box2_x1 = target_bboxes[:, 0]
    box2_y1 = target_bboxes[:, 1]
    box2_x2 = target_bboxes[:, 2]
    box2_y2 = target_bboxes[:, 3]

    x1 = np.maximum(box1_x1, box2_x1)
    # logger.info((x1, box1_x1, box2_x1))
    y1 = np.maximum(box1_y1, box2_y1)
    x2 = np.minimum(box1_x2, box2_x2)
    y2 = np.minimum(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1) * (y2 - y1)
    # area = width * height
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-16)

def precision_recall(true_boxes, pred_boxes, class_labels, iou_threshold=0.5):
    """
    Calculates recall and precision per class
    Parameters:
        pred_boxes (Numpy Array):
        specified as Nx[image_id, predicted_class_id, confidence_score, x1, y1, x2, y2]
        true_boxes (tensor): Similar as pred_boxes except all the correct ones
        class_labels(list): List of all possible Class IDs
        iou_threshold (float): threshold where predicted bboxes is correct
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """
    epsilon = 1e-16
    recall_per_class = {}
    precision_per_class = {}
    for c in range(class_labels):
        detections = pred_boxes[pred_boxes[:, 1] == c]
        ground_truths = true_boxes[true_boxes[:, 1] == c]
        amount_bboxes = Counter(ground_truths[:, 0].tolist())
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = np.zeros(val)
        detections = sorted(detections, key=lambda x: x[2], reverse=True)
        TP = np.zeros((len(detections)))
        FP = np.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = ground_truths[ground_truths[:, 0] == detection[0]]
            num_gts = len(ground_truth_img)
            if num_gts == 0:
                best_iou = 0
            else:
                ious = intersection_over_union(detection[-4:][None, :], ground_truth_img[:, -4:])
                # logger.info(ious)
                best_gt_idx = np.argmax(ious)
                best_iou = ious[best_gt_idx]
            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1
            # logger.info(best_iou)
        TP_cumsum = np.cumsum(TP, axis=0)
        FP_cumsum = np.cumsum(FP, axis=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        recall_per_class[c] = np.mean(recalls)
        precision_per_class[c] = np.mean(precisions)
    return recall_per_class, precision_per_class