import os
from loguru import logger
import evaluation
from collections import defaultdict
from evaluation import precision_recall
import numpy as np

def encode_imagenames(gt_filename):
    """
    Encodes filenames and classes from file

    Parameters:
        gt_filename: String containig path to ground truth file

    Result:
        filename_encoding: Dictionary of filename encodings
        class_encoding: Dictionary of class encodings

    """
    filename_encoding = {}
    filename_idx = 0
    class_encoding = {}
    class_idx = 0
    with open(gt_filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        filename, bbox, cls = line.split()
        if filename not in filename_encoding:
            filename_encoding[filename] = filename_idx
            filename_idx += 1
        if cls not in class_encoding:
            class_encoding[cls] = class_idx
            class_idx += 1
    return filename_encoding, class_encoding

def make_calculation(gt_filename, dt_filename, iou_threshold=0.5):
    """
    Printing the metrics (recall, precision, iou)

    Parameters:
        gt_filename: String containig path to ground truth file
        dt_filename: String containig path to detection file
        iou_threshold: Float threshold for iou to be true positive

    Result:
        None

    """
    img_encoding, class_encoding = encode_imagenames(gt_filename)
    logger.info(class_encoding)
    with open(gt_filename, "r") as f:
        gt_samples, gt_classes= txt2dict(f.readlines(), img_encoding, class_encoding)
    with open(dt_filename, "r") as f:
        dt_samples, dt_classes = txt2dict(f.readlines(), img_encoding, class_encoding)

    classes = gt_classes.union(dt_classes)

    logger.info(len(classes))
    pr_bboxes = np.array(dt_samples)
    gt_bboxes = np.array(gt_samples)
    recall_per_class, precision_per_class, iou_per_class = precision_recall(pr_bboxes, gt_bboxes, len(classes), iou_threshold)
    logger.info((recall_per_class, precision_per_class, iou_per_class))


def txt2dict(lines, img_encoding, class_encoding):
    """
    Converting the lines from the file to list of [image_id, predicted_class_id, confidence_score, x1, y1, x2, y2]

    Parameters:
        lines: Readlines from the file
        img_encoding: Dictionary of filename encodings
        class_encoding: Dictionary of class encodings

    Result:
        samples: list of [image_id, predicted_class_id, confidence_score, x1, y1, x2, y2]
        classes: set of classes used in the file
    """
    logger.info(len(lines))
    samples = []
    classes = set()
    for line in lines:
        line_split =  line.split()
        assert len(line_split) == 4 or len(line_split) == 3
        if len(line_split) == 3:
            filename, bbox, cls = line_split
            x1, y1, x2, y2 = bbox.split(",")
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            samples.append([img_encoding[filename], class_encoding[cls], 1.0, x1, y1, x2, y2])

        else:
            filename, bbox, cls, confidence = line_split
            x1, y1, x2, y2 = bbox.split(",")
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            samples.append([img_encoding[filename], class_encoding[cls], float(confidence), x1, y1, x2, y2])

        classes.add(cls)
    return samples, classes