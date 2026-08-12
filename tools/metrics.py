import torch
from torchmetrics import Metric
from torchmetrics.classification import MultilabelStatScores



class IntersectionOverUnion(Metric):
    """Computes intersection-over-union."""
    def __init__(self, n_classes: int, absent_score: float = 1.0):
        super().__init__()

        self.n_classes = n_classes
        self.absent_score = absent_score

        self.metric = MultilabelStatScores(num_labels=n_classes, 
                                           average=None,
                                           multidim_average='global')


    def update(self, prediction: torch.Tensor, target: torch.Tensor):

        # Update the metric
        self.metric.update(prediction, target)


    def compute(self):
        stats = self.metric.compute()   # shape [n_classes, [tp, fp, tn, fn]]
        tp = stats[..., 0]  # True Positives
        fp = stats[..., 1]  # False Positives
        fn = stats[..., 3]  # False Negatives
        # iou = tp / (tp + fp + fn)  # shape: [n_classes]
        denom = tp + fp + fn
        iou = torch.zeros_like(denom, dtype=torch.float32)

        valid = denom > 0
        iou[valid] = tp[valid].float() / denom[valid]
        iou[~valid] = self.absent_score

        return iou