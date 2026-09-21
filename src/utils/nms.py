# https://github.com/matin-ghorbani/Non-Max-Suppression-from-Scratch/blob/main/LICENSE

import torch

from utils.IoU import intersection_over_union

def use_nms(results, conf, iou): #must convert 'results' in a format that fits this funciton

    for result in results:
        nms_class = []
        nms_probability = []
        nms_xyxy = []
        nms_class.extend(result.boxes.cls.int().tolist())
        nms_probability.extend(result.boxes.conf.tolist())
        nms_xyxy.extend(result.boxes.xyxy.tolist())
        predictions_old = zip(nms_class, nms_probability, nms_xyxy)
        predictions =   [
                            [list_class, list_conf, list_xyxy[0], list_xyxy[1], list_xyxy[2], list_xyxy[3], i]
                            for i, (list_class, list_conf, list_xyxy) in enumerate(predictions_old)
                        ]
        bboxes_after_nms = non_max_suppression(predictions=predictions, prob_thresh=conf, iou_thresh=iou)
        results_nms = [box[-1] for box in bboxes_after_nms] # the positional index added to make the nms function work
        result.boxes = result.boxes[results_nms]

    return results

def non_max_suppression(
    predictions,
    prob_thresh,
    iou_thresh,
    box_format='corners',
):
    # Predictions: [[class, probability, x1, y1, x2, y2]]
    assert isinstance(predictions, list)

    bboxes = [
        box
        for box in predictions
        if box[1] >= prob_thresh
    ]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)  # Choose the box with the highest probability

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]  # If they don't have a same class
            or intersection_over_union(
                torch.tensor(chosen_box[2:6]),  # x1, y1, x2, y2
                torch.tensor(box[2:6]),  # x1, y1, x2, y2
                box_format=box_format
            ) < iou_thresh
        ]

        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms