import numpy as np
from sklearn.metrics import confusion_matrix
from monai.metrics import HausdorffDistanceMetric
# import torch.nn.functional as F
import torch

def compute_dice(pred, true, smooth=1e-5):
    intersection = np.sum(pred * true)
    summation = np.sum(pred) + np.sum(true)
    if summation == 0:
        return 1.0
    return 2.0 * intersection / (summation + smooth)


def evaluate(y_scores, y_true, interval=0.02):
    y_pred = (torch.softmax(y_scores.clone().detach().to(dtype=torch.float32), dim=1)[:, 1:2, ...] > 0.5).float()

    y_t = y_true.clone().detach().to(dtype=torch.float32).unsqueeze(1)

    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction='mean', percentile=95)
    hd95 = hausdorff_metric(y_pred=y_pred, y=y_t)
    hd95_np = hd95.cpu().numpy()
    valid_distances = hd95_np[~np.isnan(hd95_np) & ~np.isinf(hd95_np)]

    # 如果筛选后没有有效的距离值，则返回NaN
    # 否则，计算剩余有效距离值的平均值
    if valid_distances.size == 0:
        m_hausdorff = np.nan
    else:
        m_hausdorff = valid_distances.mean()

    y_scores = torch.softmax(y_scores, dim=1)
    y_scores = y_scores[:, 1, ...].cpu().detach().numpy().flatten()
    y_true = y_true.data.cpu().numpy().flatten()

    thresholds = np.arange(0, 0.9, interval)
    jaccard = np.zeros(len(thresholds))
    dice = np.zeros(len(thresholds))
    y_true.astype(np.int8)

    for indy in range(len(thresholds)):
        threshold = thresholds[indy]
        y_pred = (y_scores > threshold).astype(np.int8)

        sum_area = (y_pred + y_true)
        tp = float(np.sum(sum_area == 2))
        union = np.sum(sum_area == 1)
        jaccard[indy] = tp / float(union + tp)
        dice[indy] = 2 * tp / float(union + 2 * tp)

    thred_indx = np.argmax(jaccard)
    m_jaccard = jaccard[thred_indx]
    m_dice = dice[thred_indx]

    return thresholds[thred_indx], m_jaccard, m_dice,m_hausdorff

def evaluate_multi_binary_labels(y_pred_label, y_true):
    """Evaluate binary segmentation metrics from class-label volumes."""
    y_pred_np = y_pred_label.cpu().numpy().astype(np.int8)
    y_true_np = y_true.cpu().numpy().astype(np.int8)

    # 构造二值 mask，认为1代表目标（肿瘤区域），0代表背景
    pred = (y_pred_np == 1).astype(np.int8)
    true = (y_true_np == 1).astype(np.int8)

    # 计算 Dice 值，使用你已有的 compute_dice 函数
    dice = compute_dice(pred, true)

    # 计算 Hausdorff 距离（hd95）
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction='mean', percentile=95)
    pred_tensor = torch.from_numpy(pred).unsqueeze(1).float()
    true_tensor = torch.from_numpy(true).unsqueeze(1).float()
    hd95 = hausdorff_metric(y_pred=pred_tensor, y=true_tensor)
    if hd95.numel() > 1:
        hd95 = hd95.mean()
    hd95 = hd95.item()
    
    return dice, hd95

def evaluate_multi_binary(y_scores, y_true):
    """
    y_scores: 模型输出 logits，形状为 [B, 2, 128, 128, 128]
    y_true: ground truth，形状为 [B, 128, 128, 128]，取值为 {0,1}

    评估二分类分割的指标，返回 (dice, hd95)
    """
    y_pred_label = torch.argmax(torch.softmax(y_scores, dim=1), dim=1)
    return evaluate_multi_binary_labels(y_pred_label, y_true)


def evaluate_multi_labels(y_pred_label, y_true):
    """
    y_pred_label: 预测类别标签，形状为 [B, 128, 128, 128]
    y_true: ground truth，形状为 [B, 128, 128, 128]，取值在 {0,1,2,3}

    对于每个结构：
      - ET: 仅类别 3
      - TC: 类别 1 和 3
      - WT: 类别 1、2 和 3
    返回每个结构的 (dice, hd95)
    """
    y_pred_np = y_pred_label.cpu().numpy().astype(np.int8)
    y_true_np = y_true.cpu().numpy().astype(np.int8)

    # 构造各结构的二值 mask
    pred_et = (y_pred_np == 3).astype(np.int8)
    true_et = (y_true_np == 3).astype(np.int8)

    pred_tc = ((y_pred_np == 1) | (y_pred_np == 3)).astype(np.int8)
    true_tc = ((y_true_np == 1) | (y_true_np == 3)).astype(np.int8)

    pred_wt = ((y_pred_np == 1) | (y_pred_np == 2) | (y_pred_np == 3)).astype(np.int8)
    true_wt = ((y_true_np == 1) | (y_true_np == 2) | (y_true_np == 3)).astype(np.int8)

    # 计算 Dice 值
    dice_et = compute_dice(pred_et, true_et)
    dice_tc = compute_dice(pred_tc, true_tc)
    dice_wt = compute_dice(pred_wt, true_wt)

    # 计算 Hausdorff 距离（hd95）
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction='mean', percentile=95)

    # 将 numpy 二值 mask 转换为 tensor 并加上 channel 维度
    pred_et_tensor = torch.from_numpy(pred_et).unsqueeze(1).float()
    true_et_tensor = torch.from_numpy(true_et).unsqueeze(1).float()
    hd95_et = hausdorff_metric(y_pred=pred_et_tensor, y=true_et_tensor)
    # 如果 hd95_et 不是标量，则取平均
    if hd95_et.numel() > 1:
        hd95_et = hd95_et.mean()
    hd95_et = hd95_et.item()

    pred_tc_tensor = torch.from_numpy(pred_tc).unsqueeze(1).float()
    true_tc_tensor = torch.from_numpy(true_tc).unsqueeze(1).float()
    hd95_tc = hausdorff_metric(y_pred=pred_tc_tensor, y=true_tc_tensor)
    if hd95_tc.numel() > 1:
        hd95_tc = hd95_tc.mean()
    hd95_tc = hd95_tc.item()

    pred_wt_tensor = torch.from_numpy(pred_wt).unsqueeze(1).float()
    true_wt_tensor = torch.from_numpy(true_wt).unsqueeze(1).float()
    hd95_wt = hausdorff_metric(y_pred=pred_wt_tensor, y=true_wt_tensor)
    if hd95_wt.numel() > 1:
        hd95_wt = hd95_wt.mean()
    hd95_wt = hd95_wt.item()

    return dice_et, hd95_et, dice_tc, hd95_tc,dice_wt, hd95_wt


def evaluate_multi(y_scores, y_true):
    """Evaluate multiclass metrics from model logits."""
    y_pred_label = torch.argmax(torch.softmax(y_scores, dim=1), dim=1)
    return evaluate_multi_labels(y_pred_label, y_true)

def evaluate_groupwise_binary(y_scores, y_true, group_size=10):
    y_pred_label = torch.argmax(torch.softmax(y_scores, dim=1), dim=1)
    return evaluate_groupwise_binary_labels(y_pred_label, y_true, group_size)


def evaluate_groupwise_binary_labels(y_pred_label, y_true, group_size=10):
    total_samples = y_pred_label.shape[0]
    num_groups = int(np.ceil(total_samples / group_size))
    sum_metrics = np.zeros(2)  # 对应 (dice, hd95)
    sample_count = 0
    for i in range(num_groups):
        idx_start = i * group_size
        idx_end = min((i + 1) * group_size, total_samples)
        current_scores = y_pred_label[idx_start:idx_end]
        current_truth = y_true[idx_start:idx_end]
        metrics = evaluate_multi_binary_labels(current_scores, current_truth)
        group_samples = current_scores.shape[0]
        sum_metrics += np.array(metrics) * group_samples
        sample_count += group_samples
    avg_metrics = sum_metrics / sample_count
    return tuple(avg_metrics)

def evaluate_groupwise(y_scores, y_true, group_size=50):
    y_pred_label = torch.argmax(torch.softmax(y_scores, dim=1), dim=1)
    return evaluate_groupwise_labels(y_pred_label, y_true, group_size)


def evaluate_groupwise_labels(y_pred_label, y_true, group_size=50):
    """
    当样本数量较大时，将数据按 group_size 分组计算评估指标，然后对所有组的指标进行加权平均。
    返回 (dice_et, hd95_et, dice_tc, hd95_tc, dice_wt, hd95_wt)
    """
    # import numpy as np

    total_samples = y_pred_label.shape[0]
    num_groups = int(np.ceil(total_samples / group_size))
    sum_metrics = np.zeros(6)  # 累计各指标和
    sample_count = 0

    for i in range(num_groups):
        idx_start = i * group_size
        idx_end = min((i + 1) * group_size, total_samples)
        current_scores = y_pred_label[idx_start:idx_end]
        current_truth = y_true[idx_start:idx_end]
        metrics = evaluate_multi_labels(current_scores, current_truth)
        # 将当前组指标乘以当前组样本数并累加
        group_samples = current_scores.shape[0]
        sum_metrics += np.array(metrics) * group_samples
        sample_count += group_samples

    avg_metrics = sum_metrics / sample_count
    return tuple(avg_metrics)


class StreamingValidationMetrics:
    """Compute the existing groupwise validation metrics with bounded RAM use.

    Predictions are reduced to class labels before they leave the accelerator.
    At most one metric group is retained for each branch, matching the grouping
    used by ``print_val_eval`` while avoiding accumulation of full-resolution
    logits for the complete validation fold.
    """

    def __init__(self, num_classes, group_size=None):
        self.num_classes = num_classes
        # Keep the grouping used by the existing validation printout. The
        # caller can override this if a future experiment changes that policy.
        self.group_size = group_size or 10
        self._prediction_buffers = [[], []]
        self._target_buffer = []
        self._buffer_size = 0
        metric_count = 2 if num_classes == 2 else 6
        self._metric_sums = [np.zeros(metric_count), np.zeros(metric_count)]
        self._sample_count = 0

    def _record_group(self, predictions_1, predictions_2, target):
        if self.num_classes == 2:
            metric_fn = evaluate_multi_binary_labels
        else:
            metric_fn = evaluate_multi_labels

        metrics_1 = np.asarray(metric_fn(predictions_1, target), dtype=float)
        metrics_2 = np.asarray(metric_fn(predictions_2, target), dtype=float)
        group_samples = predictions_1.shape[0]
        self._metric_sums[0] += metrics_1 * group_samples
        self._metric_sums[1] += metrics_2 * group_samples
        self._sample_count += group_samples

    def _flush_full_groups(self):
        while self._buffer_size >= self.group_size:
            predictions_1 = torch.cat(self._prediction_buffers[0], dim=0)
            predictions_2 = torch.cat(self._prediction_buffers[1], dim=0)
            target = torch.cat(self._target_buffer, dim=0)

            self._record_group(
                predictions_1[:self.group_size],
                predictions_2[:self.group_size],
                target[:self.group_size],
            )

            predictions_1 = predictions_1[self.group_size:].clone()
            predictions_2 = predictions_2[self.group_size:].clone()
            target = target[self.group_size:].clone()
            self._prediction_buffers = [[predictions_1] if predictions_1.shape[0] else [],
                                        [predictions_2] if predictions_2.shape[0] else []]
            self._target_buffer = [target] if target.shape[0] else []
            self._buffer_size = target.shape[0]

    @torch.no_grad()
    def update(self, predictions_1, predictions_2, target):
        """Add one validation batch of class labels to the metric stream."""
        predictions_1 = predictions_1.detach().to(device='cpu', dtype=torch.uint8)
        predictions_2 = predictions_2.detach().to(device='cpu', dtype=torch.uint8)
        target = target.detach().to(device='cpu', dtype=torch.uint8)
        if predictions_1.shape != predictions_2.shape or predictions_1.shape != target.shape:
            raise ValueError('Validation predictions and target must have matching shapes.')

        self._prediction_buffers[0].append(predictions_1)
        self._prediction_buffers[1].append(predictions_2)
        self._target_buffer.append(target)
        self._buffer_size += target.shape[0]
        self._flush_full_groups()

    def compute(self):
        """Flush the final partial group and return both branch metric tuples."""
        if self._buffer_size:
            predictions_1 = torch.cat(self._prediction_buffers[0], dim=0)
            predictions_2 = torch.cat(self._prediction_buffers[1], dim=0)
            target = torch.cat(self._target_buffer, dim=0)
            self._record_group(predictions_1, predictions_2, target)
            self._prediction_buffers = [[], []]
            self._target_buffer = []
            self._buffer_size = 0

        if self._sample_count == 0:
            raise ValueError('Cannot compute validation metrics without samples.')
        return tuple(self._metric_sums[0] / self._sample_count), tuple(
            self._metric_sums[1] / self._sample_count
        )

# def evaluate_multi(y_scores, y_true):

#     y_scores = torch.softmax(y_scores, dim=1)
#     y_pred = torch.max(y_scores, 1)[1]
#     y_pred = y_pred.data.cpu().numpy().flatten()
#     y_true = y_true.data.cpu().numpy().flatten()

#     hist = confusion_matrix(y_true, y_pred)

#     hist_diag = np.diag(hist)
#     hist_sum_0 = hist.sum(axis=0)
#     hist_sum_1 = hist.sum(axis=1)

#     jaccard = hist_diag / (hist_sum_1 + hist_sum_0 - hist_diag)
#     m_jaccard = np.nanmean(jaccard)
#     dice = 2 * hist_diag / (hist_sum_1 + hist_sum_0)
#     m_dice = np.nanmean(dice)

#     return jaccard, m_jaccard, dice, m_dice

