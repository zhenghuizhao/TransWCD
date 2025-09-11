import numpy as np
import sklearn.metrics as metrics



def _fast_hist(label_true, label_pred, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask],
        minlength=num_classes ** 2,
    )
    return hist.reshape(num_classes, num_classes)

def scores(label_trues, label_preds, num_classes=2):
    hist = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_trues, label_preds):
        # hist is confusion_matrix
        hist += _fast_hist(lt.flatten(), lp.flatten(), num_classes)
        # tp = np.diag(hist)
        # sum_a1 = hist.sum(axis=1)
        # sum_a0 = hist.sum(axis=0)

    # OA
    acc = np.diag(hist).sum() / (hist.sum() + np.finfo(np.float32).eps)  # acc = np.diag(hist).sum() / hist.sum()

    # recall
    recall = np.diag(hist) / (hist.sum(axis=1) + np.finfo(np.float32).eps)  # np.finfo(np.float32).eps
    # acc_cls = np.nanmean(recall)

    # precision
    precision = np.diag(hist) / (hist.sum(axis=0) + np.finfo(np.float32).eps )

    # F1 score
    F1 = 2 * recall * precision / (recall + precision+ np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # return mean_F1

    # IoU
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + np.finfo(np.float32).eps )
    valid = hist.sum(axis=1) > 0  # added
    # mean_iu = np.nanmean(iu[valid])
    # freq = hist.sum(axis=1) / (hist.sum())
    # cls_iu = dict(zip(range(num_classes), iu))


###
    cls_iu = dict(zip(range(num_classes), iu))
    cls_precision = dict(zip(range(num_classes), precision))
    cls_recall = dict(zip(range(num_classes), recall))
    cls_F1 = dict(zip(range(num_classes), F1))


    score_dict = {'OA': acc, 'f1': cls_F1, 'precision': cls_precision, 'iou': cls_iu, 'recall': cls_recall}

    return score_dict