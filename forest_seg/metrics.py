import numpy as np


def confusion(pred, y):
    p, y = np.asarray(pred).astype(bool), np.asarray(y).astype(bool)
    return int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum())


def scores(tp, fp, fn, tn):
    iou_f = tp / max(tp + fp + fn, 1)
    iou_n = tn / max(tn + fn + fp, 1)
    return dict(
        iou_forest=iou_f,
        iou_nonforest=iou_n,
        miou=(iou_f + iou_n) / 2,
        dice_forest=2 * tp / max(2 * tp + fp + fn, 1),
        acc=(tp + tn) / max(tp + fp + fn + tn, 1),
    )
