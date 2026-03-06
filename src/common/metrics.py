# Some sections of this code reused code from SemanticKITTI development kit
# https://github.com/PRBonn/semantic-kitti-api


import numpy as np
import torch


class IoUEval:
    def __init__(self, n_classes: int, ignore=(), only_present_in_mean: bool = True):
        self.n_classes = int(n_classes)
        self.ignore = np.array(ignore, dtype=np.int64).reshape(-1)
        self.include = np.array(
            [c for c in range(self.n_classes) if c not in set(self.ignore)],
            dtype=np.int64,
        )
        self.only_present_in_mean = bool(only_present_in_mean)
        self.reset()

    def num_classes(self) -> int:
        return self.n_classes

    def reset(self):
        self.conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def add_batch(self, x, y):
        # x: preds, y: targets
        x = np.asarray(x)
        y = np.asarray(y)
        assert x.shape == y.shape, "preds and targets must have same shape"

        x_row = x.reshape(-1).astype(np.int64)
        y_row = y.reshape(-1).astype(np.int64)
        assert x_row.shape == y_row.shape

        # Optional: clamp/validate indices
        valid = (
            (x_row >= 0)
            & (x_row < self.n_classes)
            & (y_row >= 0)
            & (y_row < self.n_classes)
        )
        if not np.all(valid):
            x_row = x_row[valid]
            y_row = y_row[valid]

        np.add.at(self.conf_matrix, (x_row, y_row), 1)

    def _conf_after_ignore(self):
        conf = self.conf_matrix.copy()
        if self.ignore.size > 0:
            conf[:, self.ignore] = 0  # drop pixels whose GT is ignored
        return conf

    def get_stats(self):
        conf = self._conf_after_ignore()
        tp = np.diag(conf)
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def get_iou(self):
        tp, fp, fn = self.get_stats()
        union = tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, tp / union, np.nan)

        # mean over include
        include_mask = np.isin(np.arange(self.n_classes), self.include)
        if self.only_present_in_mean:
            present = union > 0
            mean_mask = include_mask & present
        else:
            mean_mask = include_mask

        if np.any(mean_mask):
            iou_mean = np.nanmean(iou[mean_mask])
        else:
            iou_mean = float("nan")

        # return per-class IoU with NaNs for absent classes (more informative than hard 0)
        return iou_mean, iou

    def get_pixel_accuracy(self):
        # Overall pixel accuracy after ignoring GT columns
        conf = self._conf_after_ignore()
        correct = np.trace(conf)
        total = conf.sum()
        return float(correct) / (float(total) + 1e-15)

    def get_mean_class_accuracy(self):
        # Mean of per-class recall: tp / (tp + fn)
        tp, _, fn = self.get_stats()
        denom = tp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.where(denom > 0, tp / denom, np.nan)
        mask = np.isin(np.arange(self.n_classes), self.include) & (denom > 0)
        return np.nanmean(acc[mask]) if np.any(mask) else float("nan")

    def get_confusion(self):
        return self.conf_matrix.copy()


class LossesTrackEpoch:
    def __init__(self, num_iterations):
        # classes
        self.num_iterations = num_iterations
        self.validation_losses = {}
        self.train_losses = {}
        self.train_iteration_counts = 0
        self.validation_iteration_counts = 0

    def set_validation_losses(self, keys):
        for key in keys:
            self.validation_losses[key] = 0

    def set_train_losses(self, keys):
        for key in keys:
            self.train_losses[key] = 0

    def update_train_losses(self, loss):
        for key in loss:
            self.train_losses[key] += loss[key]
        self.train_iteration_counts += 1

    def update_validaiton_losses(self, loss):
        for key in loss:
            self.validation_losses[key] += loss[key]
        self.validation_iteration_counts += 1

    def restart_train_losses(self):
        for key in self.train_losses:
            self.train_losses[key] = 0
        self.train_iteration_counts = 0

    def restart_validation_losses(self):
        for key in self.validation_losses:
            self.validation_losses[key] = 0
        self.validation_iteration_counts = 0


class Metrics:

    def __init__(self, nbr_classes, num_iterations_epoch, scales):

        self.nbr_classes = nbr_classes
        self.evaluator = {}
        for scale in scales:
            self.evaluator[scale] = IoUEval(self.nbr_classes, [])
        self.losses_track = LossesTrackEpoch(num_iterations_epoch)
        self.best_metric_record = {"mIoU": 0, "IoU": 0, "epoch": 0, "loss": 99999999}

    def add_batch(self, prediction, target):

        # passing to cpu
        for key in prediction:
            prediction[key] = torch.argmax(prediction[key], dim=1).data.cpu().numpy()
        for key in target:
            target[key] = target[key].data.cpu().numpy()

        for key in target:
            prediction["pred_semantic_" + key] = (
                prediction["pred_semantic_" + key].reshape(-1).astype("int64")
            )
            target[key] = target[key].reshape(-1).astype("int64")
            lidar_mask = self.get_eval_mask_lidar(target[key])
            self.evaluator[key].add_batch(
                prediction["pred_semantic_" + key][lidar_mask], target[key][lidar_mask]
            )

    @staticmethod
    def get_eval_mask_lidar(target):
        """
        eval_mask_lidar is only to ingore unknown voxels in groundtruth
        """
        mask = target != 255
        return mask

    def get_occupancy_iou(self, scale):
        conf = self.evaluator[scale].get_confusion()
        tp_occupancy = np.sum(conf[1:, 1:])
        fp_occupancy = np.sum(conf[1:, 0])
        fn_occupancy = np.sum(conf[0, 1:])
        intersection = tp_occupancy
        union = tp_occupancy + fp_occupancy + fn_occupancy + 1e-15
        iou_occupancy = intersection / union
        return iou_occupancy  # returns iou occupancy

    def get_occupancy_precision(self, scale):
        conf = self.evaluator[scale].get_confusion()
        tp_occupancy = np.sum(conf[1:, 1:])
        fp_occupancy = np.sum(conf[1:, 0])
        precision = tp_occupancy / (tp_occupancy + fp_occupancy + 1e-15)
        return precision  # returns precision occupancy

    def get_occupancy_recall(self, scale):
        conf = self.evaluator[scale].get_confusion()
        tp_occupancy = np.sum(conf[1:, 1:])
        fn_occupancy = np.sum(conf[0, 1:])
        recall = tp_occupancy / (tp_occupancy + fn_occupancy + 1e-15)
        return recall  # returns recall occupancy

    def get_occupancy_f1(self, scale):
        conf = self.evaluator[scale].get_confusion()
        tp_occupancy = np.sum(conf[1:, 1:])
        fn_occupancy = np.sum(conf[0, 1:])
        fp_occupancy = np.sum(conf[1:, 0])
        precision = tp_occupancy / (tp_occupancy + fp_occupancy + 1e-15)
        recall = tp_occupancy / (tp_occupancy + fn_occupancy + 1e-15)
        F1 = 2 * (precision * recall) / (precision + recall + 1e-15)
        return F1  # returns recall occupancy

    def get_semantics_miou(self, scale):
        _, class_jaccard = self.evaluator[scale].get_iou()
        mIoU_semantics = class_jaccard[1:].mean()  # Ignore on free voxels (0 excluded)
        return mIoU_semantics  # returns mIoU semantics

    def reset_evaluator(self):
        for key in self.evaluator:
            self.evaluator[key].reset()

    def update_best_metric_record(self, mIoU, IoU, loss, epoch):
        self.best_metric_record["mIoU"] = mIoU
        self.best_metric_record["IoU"] = IoU
        self.best_metric_record["loss"] = loss
        self.best_metric_record["epoch"] = epoch
