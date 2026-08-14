import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import time
import os
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import wandb
import nibabel as nib

from config.train_test_config.train_test_config import (
    print_val_eval_metrics,
    print_val_jaccard_metrics,
    print_val_loss,
)
from config.eval_config.eval import StreamingValidationMetrics
from config.warmup_config.warmup import GradualWarmupScheduler
from loss.loss_function import segmentation_loss
from model.HFF import HFFNet
from loader.dataload3d import get_loaders
from utils.utils import get_device
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


def init_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_label_mapping(dataset_name, class_type):
    if dataset_name in ('brats19','brats20','msdbts'):
        raw = [0,1,2,4]
    else:
        raw = [0,1,2,3]
    if class_type == 'et':
        pos = raw[-1]
        mapping = {l:(1 if l==pos else 0) for l in raw}
    elif class_type == 'tc':
        p1, p2 = 1, raw[-1]
        mapping = {l:(1 if l in (p1,p2) else 0) for l in raw}
    elif class_type == 'wt':
        p2, p3 = 2, raw[-1]
        mapping = {l:(1 if l in (1,p2,p3) else 0) for l in raw}
    else:
        mapping = { old:new for new, old in enumerate(raw) }
    return mapping


def mask_to_class_indices(mask, mapping):
    out = torch.zeros_like(mask, dtype=torch.long)
    for old, new in mapping.items():
        out[mask == old] = new
    return out


def add_evaluation_arguments(parser, include_checkpoint=True, include_output_dir=True):
    """Add the inference arguments shared by ``eval.py`` and ``cross_eval.py``."""
    parser.add_argument(
        '--test_list',
        type=str,
        help='Path to text file listing test volumes',
        default='./bratsxxx/x-val.txt',
    )
    if include_checkpoint:
        parser.add_argument(
            '--checkpoint',
            type=str,
            help='Path to trained model checkpoint (.pth)',
            default='yourpath/your_trained_hff-net.pth',
        )
    parser.add_argument('--dataset_name', choices=['brats19','brats20','brats23men','msdbts'], default='brats19')
    parser.add_argument('--class_type', choices=['all'], default='all')
    parser.add_argument('--selected_modal', nargs='+', default=[
        'flair_L','t1_L','t1ce_L','t2_L',
        'flair_H1','flair_H2','flair_H3','flair_H4',
        't1_H1','t1_H2','t1_H3','t1_H4',
        't1ce_H1','t1ce_H2','t1ce_H3','t1ce_H4',
        't2_H1','t2_H2','t2_H3','t2_H4'], help='Modalities')
    # Brats 23
    # parser.add_argument('--selected_modal', nargs='+',
    #                     default=['t2w_L', 't1n_L', 't1c_L', 't2f_L', 't2f_H1', 't2f_H2', 't2f_H3', 't2f_H4',
    #                              't1n_H1', 't1n_H2', 't1n_H3', 't1n_H4', 't1c_H1',
    #                              't1c_H2', 't1c_H3', 't1c_H4', 't2w_H1', 't2w_H2', 't2w_H3', 't2w_H4'])
    
    
    parser.add_argument('-b','--batch_size', type=int, default=1)
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='DataLoader worker processes for inference (default: 8).',
    )
    parser.add_argument('-l','--loss', type=str, default='dice')
    parser.add_argument('--loss2', type=str, default='ff')
    if include_output_dir:
        parser.add_argument('--output_dir', type=str, default='./result/eval')


def build_parser():
    parser = argparse.ArgumentParser(description='HFF-Net 3D Inference')
    add_evaluation_arguments(parser)
    return parser


def evaluate_checkpoint(args, checkpoint_path=None, device=None):
    """Run the existing evaluation loop for one checkpoint and return JSON-ready metrics.

    ``cross_eval.py`` calls this function once per checkpoint. Keeping the model
    construction, data loading, loss calculation, and metric accumulation here
    ensures single-checkpoint and multi-checkpoint evaluation use identical
    inference behaviour.
    """
    checkpoint_path = checkpoint_path or args.checkpoint
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint_path}')

    test_list = Path(args.test_list).expanduser().resolve()
    if not test_list.is_file():
        raise FileNotFoundError(f'Test list does not exist: {test_list}')

    if device is None:
        device = get_device()

    init_seeds(42)
    os.makedirs(args.output_dir, exist_ok=True)

    mapping = make_label_mapping(args.dataset_name, args.class_type)
    classnum = 4 if args.class_type == 'all' else 2
    criterion = segmentation_loss(args.loss, False, cn=classnum).to(device)

    # Keep checkpoint loading identical to the original eval.py implementation.
    model = HFFNet(4, 16, classnum).to(device)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # Loader: using the same list for train and val since get_loaders expects both.
    data_files = dict(train=str(test_list), val=str(test_list))
    loaders = get_loaders(
        data_files,
        args.selected_modal,
        args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = loaders['val']
    num_batches = len(val_loader)
    if num_batches == 0:
        raise ValueError(f'Test list contains no evaluation samples: {test_list}')

    val_loss_sup_1 = 0.0
    val_loss_sup_2 = 0.0

    # Inference. This is the original eval.py loop, extracted without changing
    # preprocessing, branch outputs, or the StreamingValidationMetrics policy.
    validation_metrics = StreamingValidationMetrics(classnum, group_size=10)
    with torch.inference_mode():
        for data in tqdm(val_loader, desc=f'Inference [{checkpoint_path.name}]'):
            low_freq_inputs = []
            high_freq_inputs = []
            for j in range(20):
                tensor = data[j].unsqueeze(1).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                if j in [0, 1, 2, 3]:
                    low_freq_inputs.append(tensor)
                else:
                    high_freq_inputs.append(tensor)
            low = torch.cat(low_freq_inputs, dim=1)
            high = torch.cat(high_freq_inputs, dim=1)
            mask_val = mask_to_class_indices(data[20], mapping).long().to(
                device,
                non_blocking=True,
            )

            outputs_val_1, outputs_val_2, side1, side2 = model(low, high)
            loss1 = criterion(outputs_val_1, mask_val)
            loss2 = criterion(outputs_val_2, mask_val)
            pred_val_1 = torch.argmax(outputs_val_1, dim=1)
            pred_val_2 = torch.argmax(outputs_val_2, dim=1)
            validation_metrics.update(pred_val_1, pred_val_2, mask_val)

            val_loss_sup_1 += loss1.item()
            val_loss_sup_2 += loss2.item()
            del (low, high, mask_val, outputs_val_1, outputs_val_2,
                 side1, side2, loss1, loss2, pred_val_1, pred_val_2)

    print_val_loss(val_loss_sup_1, val_loss_sup_2, {'val': num_batches}, 63, 0)
    val_eval_list_1, val_eval_list_2 = validation_metrics.compute()
    val_jaccard_list_1, val_jaccard_list_2 = validation_metrics.compute_jaccard()
    print_val_eval_metrics(classnum, val_eval_list_1, val_eval_list_2, 31)
    print_val_jaccard_metrics(classnum, val_jaccard_list_1, val_jaccard_list_2, 31)

    return {
        'checkpoint': str(checkpoint_path),
        'test_list': str(test_list),
        'dataset_name': args.dataset_name,
        'class_type': args.class_type,
        'num_batches': num_batches,
        'validation_loss_branch_1': val_loss_sup_1 / num_batches,
        'validation_loss_branch_2': val_loss_sup_2 / num_batches,
        'metrics': {
            'branch_1': list(val_eval_list_1),
            'branch_2': list(val_eval_list_2),
        },
        'jaccard': {
            'branch_1': list(val_jaccard_list_1),
            'branch_2': list(val_jaccard_list_2),
        },
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    evaluate_checkpoint(args)


if __name__ == '__main__':
    main()
