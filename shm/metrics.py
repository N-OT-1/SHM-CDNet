"""Global foreground metrics accumulated from integer confusion counts."""
import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS


def confusion_counts(pred, target):
    pred, target = np.asarray(pred).squeeze(), np.asarray(target).squeeze()
    if pred.shape != target.shape:
        raise ValueError('Prediction and target dimensions differ')
    valid = np.isin(target, [0, 1])
    positive, changed = pred == 1, target == 1
    return np.array([np.sum(positive & changed & valid), np.sum(positive & ~changed & valid),
                     np.sum(~positive & changed & valid), np.sum(~positive & ~changed & valid)], dtype=np.int64)


def foreground_metrics(counts):
    tp, fp, fn, tn = map(int, counts)
    def div(x, y): return x / y if y else 0.0
    return dict(f1=div(2*tp, 2*tp+fp+fn), iou=div(tp, tp+fp+fn),
                precision=div(tp, tp+fp), recall=div(tp, tp+fn),
                oa=div(tp+tn, tp+fp+fn+tn), fpr=div(fp, fp+tn),
                tp=tp, fp=fp, fn=fn, tn=tn)


@METRICS.register_module()
class ChangeMetric(BaseMetric):
    default_prefix = 'change'

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            pred = sample['pred_sem_seg']['data'].detach().cpu().numpy()
            target = sample['gt_sem_seg']['data'].detach().cpu().numpy()
            self.results.append(confusion_counts(pred, target))

    def compute_metrics(self, results):
        return foreground_metrics(np.sum(results, axis=0))
