import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

import Utils as U
from core.utils.utils import InputPadder
from scripts.eval_booster import (
    apply_protocol_defaults,
    downsample_map_linear,
    downsample_mask_nearest,
    infer_single_pair,
)


@dataclass
class BoosterSample:
    setup: str
    scene: str
    left_file: str
    right_file: str
    gt_file: str
    left_name: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, type=str)
    parser.add_argument('--booster_root', required=True, type=str)
    parser.add_argument('--split_file', default=f'{code_dir}/../configs/booster_scene_split_v1.yaml', type=str)
    parser.add_argument('--out_dir', required=True, type=str)
    parser.add_argument('--paper_protocol', default='booster_q', choices=['legacy', 'booster_full', 'booster_q'])
    parser.add_argument('--balanced_input_scale', default=None, type=float)
    parser.add_argument('--unbalanced_input_scale', default=None, type=float)
    parser.add_argument('--scale', default=1.0, type=float)
    parser.add_argument('--match_unbalanced_left_to_right', default=1, type=int)
    parser.add_argument('--benchmark_gt_resolution', default=None, choices=['full', 'quarter'])
    parser.add_argument('--hiera', default=0, type=int)
    parser.add_argument('--valid_iters', default=None, type=int)
    parser.add_argument('--max_disp', default=None, type=int)
    parser.add_argument('--low_memory', default=None, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--log_every', default=20, type=int)
    parser.add_argument('--save_every', default=1, type=int)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--amp_dtype', default='bf16', choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument('--use_uncertainty_update_gate', default=0, type=int)
    parser.add_argument('--uncertainty_gate_scale', default=1.0, type=float)
    parser.add_argument('--uncertainty_gate_bias', default=0.0, type=float)
    parser.add_argument('--uncertainty_gate_hidden_dim', default=64, type=int)
    parser.add_argument('--use_loslite_refinement', default=1, type=int)
    parser.add_argument('--loslite_hidden_dim', default=64, type=int)
    parser.add_argument('--loslite_uncertainty_margin', default=0.1, type=float)
    parser.add_argument('--loslite_propagation_blend', default=1.0, type=float)
    parser.add_argument('--loslite_grad_scale', default=0.25, type=float)
    parser.add_argument('--loslite_offset_scale', default=0.25, type=float)
    parser.add_argument('--loslite_alpha_init', default=-5.0, type=float)
    parser.add_argument('--loslite_max_residual', default=1.0, type=float)
    parser.add_argument('--loslite_preserve_weight', default=0.02, type=float)
    parser.add_argument('--sequence_loss_gamma', default=0.9, type=float)
    parser.add_argument('--train_setups', nargs='+', default=['balanced', 'unbalanced'])
    parser.add_argument('--val_setups', nargs='+', default=['balanced', 'unbalanced'])
    parser.add_argument('--train_scope', default='loslite_only', choices=['loslite_only', 'all'])
    return parser.parse_args()


def read_rgb(path):
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    return img[..., :3]


def load_split(split_file):
    with open(split_file, 'r') as f:
        split_cfg = yaml.safe_load(f)
    return split_cfg['train_scenes'], split_cfg['val_scenes'], split_cfg


def resolve_split_root(booster_root, split_dir_name='train'):
    candidates = [
        os.path.join(booster_root, split_dir_name),
        os.path.join(booster_root, split_dir_name.capitalize()),
        booster_root,
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f'Cannot find Booster split root under {booster_root}')


def collect_samples(split_root, setups, scene_names):
    samples = []
    for setup in setups:
        for scene in scene_names:
            scene_dir = os.path.join(split_root, setup, scene)
            left_dir = os.path.join(scene_dir, 'camera_00')
            right_dir = os.path.join(scene_dir, 'camera_02' if setup == 'balanced' else 'camera_01')
            gt_file = os.path.join(scene_dir, 'disp_00.npy')
            for left_file in sorted(Path(left_dir).glob('im*.png')):
                left_name = left_file.name
                right_file = os.path.join(right_dir, left_name)
                if not os.path.isfile(right_file):
                    raise FileNotFoundError(right_file)
                samples.append(BoosterSample(
                    setup=setup,
                    scene=scene,
                    left_file=str(left_file),
                    right_file=right_file,
                    gt_file=gt_file,
                    left_name=left_name,
                ))
    return samples


class BoosterTrainDataset(Dataset):
    def __init__(self, samples, args):
        self.samples = samples
        self.args = args

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img0 = read_rgb(sample.left_file)
        img1 = read_rgb(sample.right_file)
        gt = np.load(sample.gt_file).astype(np.float32)
        valid = np.isfinite(gt) & (gt > 0.0)
        ori_w = img0.shape[1]

        if sample.setup == 'unbalanced' and self.args.match_unbalanced_left_to_right and img1.shape[:2] != img0.shape[:2]:
            target_h, target_w = img1.shape[:2]
            img0 = cv2.resize(img0, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (target_w, target_h), interpolation=cv2.INTER_LINEAR) * (target_w / float(ori_w))
            valid = cv2.resize(valid.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
        elif img1.shape[:2] != img0.shape[:2]:
            raise ValueError(f'Size mismatch: {img0.shape[:2]} vs {img1.shape[:2]}')

        setup_scale = self.args.balanced_input_scale if sample.setup == 'balanced' else self.args.unbalanced_input_scale
        total_scale = float(self.args.scale) * float(setup_scale)
        if total_scale != 1.0:
            new_w = max(1, int(round(img0.shape[1] * total_scale)))
            new_h = max(1, int(round(img0.shape[0] * total_scale)))
            img0 = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_LINEAR) * total_scale
            valid = cv2.resize(valid.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0

        tensor0 = torch.from_numpy(img0).permute(2, 0, 1).float()
        tensor1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        gt = torch.from_numpy(gt).float().unsqueeze(0)
        valid = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)

        return {
            'left': tensor0,
            'right': tensor1,
            'disp': gt,
            'valid': valid,
            'setup': sample.setup,
            'scene': sample.scene,
            'name': sample.left_name,
            'left_file': sample.left_file,
            'right_file': sample.right_file,
            'gt_file': sample.gt_file,
        }


def collate_fn(batch):
    assert len(batch) == 1, 'Current training script assumes batch_size=1 for variable image sizes.'
    return batch[0]


def masked_l1(pred, gt, valid):
    valid = valid > 0.5
    if valid.sum() == 0:
        return pred.new_tensor(0.0)
    return (pred[valid] - gt[valid]).abs().mean()


def masked_sequence_l1(preds, gt, valid, gamma=0.9):
    if len(preds) == 0:
        return gt.new_tensor(0.0)
    loss = gt.new_tensor(0.0)
    num_preds = len(preds)
    for idx, pred in enumerate(preds):
        weight = float(gamma) ** float(num_preds - idx - 1)
        loss = loss + weight * masked_l1(pred, gt, valid)
    return loss


def compute_formal_booster_metrics(model, batch, args):
    setup = batch['setup']
    pred_disp, pred_meta = infer_single_pair(
        model,
        args,
        setup,
        batch['left_file'],
        batch['right_file'],
        return_native=True,
    )
    gt_disp = np.load(batch['gt_file']).astype(np.float32)
    valid_mask = np.isfinite(gt_disp) & (gt_disp > 0.0)

    if args.benchmark_gt_resolution == 'quarter':
        gt_metric = downsample_map_linear(gt_disp, 0.25, value_scale=0.25)
        valid_mask_metric = downsample_mask_nearest(valid_mask, 0.25) > 0
        if setup == 'balanced' and abs(float(pred_meta['total_scale']) - 0.25) < 1e-6:
            pred_metric = pred_meta['native_disp']
        else:
            pred_metric = downsample_map_linear(pred_disp, 0.25, value_scale=0.25)
    else:
        gt_metric = gt_disp
        pred_metric = pred_disp
        valid_mask_metric = valid_mask

    metric_errors = np.abs(pred_metric - gt_metric)[valid_mask_metric]
    return {
        'num_pixels': int(metric_errors.size),
        'bad2_count': int((metric_errors > 2.0).sum()),
        'mae_sum': float(metric_errors.sum()),
    }


def run_validation(model, loader, args):
    model.eval()
    totals = {
        'loss_sum': 0.0,
        'num_batches': 0,
        'num_pixels': 0,
        'bad2_count': 0,
        'mae_sum': 0.0,
    }
    with torch.no_grad():
        for batch in loader:
            left = batch['left'].cuda(non_blocking=True).unsqueeze(0)
            right = batch['right'].cuda(non_blocking=True).unsqueeze(0)
            gt = batch['disp'].cuda(non_blocking=True).unsqueeze(0)
            valid = batch['valid'].cuda(non_blocking=True).unsqueeze(0)
            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left, right = padder.pad(left, right)
            with torch.amp.autocast('cuda', enabled=bool(args.amp_enabled), dtype=U.AMP_DTYPE):
                pred = model.forward(left, right, iters=args.valid_iters, test_mode=True, low_memory=bool(args.low_memory), optimize_build_volume='pytorch1')
            pred = padder.unpad(pred.float())
            loss = masked_l1(pred, gt, valid)
            totals['loss_sum'] += float(loss.item())
            totals['num_batches'] += 1
            formal_metrics = compute_formal_booster_metrics(model, batch, args)
            totals['num_pixels'] += formal_metrics['num_pixels']
            totals['bad2_count'] += formal_metrics['bad2_count']
            totals['mae_sum'] += formal_metrics['mae_sum']
    return {
        'loss': totals['loss_sum'] / max(totals['num_batches'], 1),
        'bad2': totals['bad2_count'] * 100.0 / max(totals['num_pixels'], 1),
        'mae': totals['mae_sum'] / max(totals['num_pixels'], 1),
        'num_pixels': totals['num_pixels'],
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, args, split_cfg):
    ckpt = {
        'epoch': epoch,
        'best_metric': best_metric,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'args': vars(args),
        'split_cfg': split_cfg,
    }
    torch.save(ckpt, path)


def main():
    args = parse_args()
    U.set_logging_format()
    U.set_seed(args.seed)
    torch.autograd.set_grad_enabled(True)
    U.set_amp_dtype(args.amp_dtype)
    args.amp_enabled = bool(args.amp_dtype != 'fp32')

    split_root = resolve_split_root(args.booster_root, 'train')
    train_scenes, val_scenes, split_cfg = load_split(args.split_file)

    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)

    model_default_low_memory = None
    model_default_max_disp = None
    if hasattr(model, 'args') and model.args is not None:
        model_default_low_memory = getattr(model.args, 'low_memory', None)
        model_default_max_disp = getattr(model.args, 'max_disp', None)

    cfg = apply_protocol_defaults({
        'paper_protocol': args.paper_protocol,
        'scale': args.scale,
        'balanced_input_scale': args.balanced_input_scale,
        'unbalanced_input_scale': args.unbalanced_input_scale,
        'benchmark_gt_resolution': args.benchmark_gt_resolution,
        'low_memory': args.low_memory if args.low_memory is not None else model_default_low_memory,
        'max_disp': args.max_disp if args.max_disp is not None else model_default_max_disp,
    })
    args.balanced_input_scale = cfg['balanced_input_scale']
    args.unbalanced_input_scale = cfg['unbalanced_input_scale']
    args.benchmark_gt_resolution = cfg['benchmark_gt_resolution']
    args.low_memory = cfg['low_memory']
    args.max_disp = cfg['max_disp']

    model.args.valid_iters = args.valid_iters or int(model.args.valid_iters)
    args.valid_iters = int(model.args.valid_iters)
    model.args.max_disp = int(args.max_disp)
    model.args.low_memory = int(args.low_memory)
    model.args.mixed_precision = bool(args.amp_enabled)
    model.args.use_uncertainty_update_gate = bool(args.use_uncertainty_update_gate)
    model.args.uncertainty_gate_scale = float(args.uncertainty_gate_scale)
    model.args.uncertainty_gate_bias = float(args.uncertainty_gate_bias)
    model.args.uncertainty_gate_hidden_dim = int(args.uncertainty_gate_hidden_dim)
    model.args.use_loslite_refinement = bool(args.use_loslite_refinement)
    model.args.loslite_hidden_dim = int(args.loslite_hidden_dim)
    model.args.loslite_uncertainty_margin = float(args.loslite_uncertainty_margin)
    model.args.loslite_propagation_blend = float(args.loslite_propagation_blend)
    model.args.loslite_grad_scale = float(args.loslite_grad_scale)
    model.args.loslite_offset_scale = float(args.loslite_offset_scale)
    model.args.loslite_alpha_init = float(args.loslite_alpha_init)
    model.args.loslite_max_residual = float(args.loslite_max_residual)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'train_args.yaml'), 'w') as f:
        yaml.safe_dump(vars(args), f, sort_keys=True)
    with open(os.path.join(args.out_dir, 'split_used.yaml'), 'w') as f:
        yaml.safe_dump(split_cfg, f, sort_keys=False)

    model.cuda()
    if args.train_scope == 'loslite_only':
        model.freeze_all_but_loslite()
    elif args.train_scope == 'all':
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f'Unknown train_scope: {args.train_scope}')

    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_summary = {
        'trainable_num_params': int(sum(x[1] for x in trainable)),
        'frozen_num_params': int(frozen),
        'trainable_param_names': [x[0] for x in trainable],
    }
    with open(os.path.join(args.out_dir, 'trainable_summary.json'), 'w') as f:
        json.dump(trainable_summary, f, indent=2)
    logging.info(f'train scenes: {train_scenes}')
    logging.info(f'val scenes: {val_scenes}')
    logging.info(f'amp dtype: {args.amp_dtype} (enabled={args.amp_enabled})')
    logging.info(f'sequence loss gamma: {args.sequence_loss_gamma}')
    logging.info(f'trainable params: {trainable_summary["trainable_num_params"]}')
    logging.info(f'frozen params: {trainable_summary["frozen_num_params"]}')
    logging.info(f'trainable param names: {trainable_summary["trainable_param_names"]}')

    train_samples = collect_samples(split_root, args.train_setups, train_scenes)
    val_samples = collect_samples(split_root, args.val_setups, val_scenes)
    logging.info(f'train samples: {len(train_samples)}')
    logging.info(f'val samples: {len(val_samples)}')

    train_loader = DataLoader(BoosterTrainDataset(train_samples, args), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(BoosterTrainDataset(val_samples, args), batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler('cuda', enabled=bool(args.amp_enabled and U.AMP_DTYPE == torch.float16))

    start_epoch = 0
    best_metric = math.inf
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
        if args.train_scope == 'loslite_only':
            model.freeze_all_but_loslite()
        else:
            for param in model.parameters():
                param.requires_grad = True
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt['scheduler'] is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = int(ckpt['epoch']) + 1
        best_metric = float(ckpt['best_metric'])

    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_disp_loss = 0.0
        epoch_preserve = 0.0
        for step, batch in enumerate(train_loader, start=1):
            left = batch['left'].cuda(non_blocking=True).unsqueeze(0)
            right = batch['right'].cuda(non_blocking=True).unsqueeze(0)
            gt = batch['disp'].cuda(non_blocking=True).unsqueeze(0)
            valid = batch['valid'].cuda(non_blocking=True).unsqueeze(0)
            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left, right = padder.pad(left, right)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=bool(args.amp_enabled), dtype=U.AMP_DTYPE):
                _, preds = model.forward(left, right, iters=args.valid_iters, test_mode=False, low_memory=bool(args.low_memory), optimize_build_volume='pytorch1')
                preds = [padder.unpad(pred) for pred in preds]
                disp_loss = masked_sequence_l1(preds, gt, valid, gamma=args.sequence_loss_gamma)
                preserve_reg = getattr(model, 'last_loslite_regularizer', None)
                if preserve_reg is None:
                    preserve_reg = disp_loss.new_tensor(0.0)
                loss = disp_loss + float(args.loslite_preserve_weight) * preserve_reg
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            epoch_disp_loss += float(disp_loss.item())
            epoch_preserve += float(preserve_reg.item())
            if step % args.log_every == 0:
                logging.info(
                    f'epoch={epoch} step={step}/{len(train_loader)} '
                    f'loss={loss.item():.6f} disp_loss={disp_loss.item():.6f} '
                    f'preserve_reg={preserve_reg.item():.6f}'
                )

        scheduler.step()
        train_loss = epoch_loss / max(len(train_loader), 1)
        train_disp_loss = epoch_disp_loss / max(len(train_loader), 1)
        train_preserve_reg = epoch_preserve / max(len(train_loader), 1)
        # Full-model finetuning can leave large cached allocations after training.
        # Clear them before validation so the formal eval-style forward fits reliably.
        torch.cuda.empty_cache()
        val_metrics = run_validation(model, val_loader, args)
        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_disp_loss': train_disp_loss,
            'train_preserve_reg': train_preserve_reg,
            'val_loss': val_metrics['loss'],
            'val_bad2': val_metrics['bad2'],
            'val_bad2_formal': val_metrics['bad2'],
            'val_mae': val_metrics['mae'],
            'val_mae_formal': val_metrics['mae'],
            'val_metric_gt_resolution': args.benchmark_gt_resolution,
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(record)
        logging.info(
            f"epoch={epoch} train_loss={train_loss:.6f} train_disp_loss={train_disp_loss:.6f} "
            f"train_preserve_reg={train_preserve_reg:.6f} val_loss={val_metrics['loss']:.6f} "
            f"val_bad2_formal={val_metrics['bad2']:.4f} val_mae_formal={val_metrics['mae']:.6f} "
            f"metric_gt_resolution={args.benchmark_gt_resolution}"
        )

        with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        if val_metrics['bad2'] < best_metric:
            best_metric = val_metrics['bad2']
            save_checkpoint(os.path.join(args.out_dir, 'checkpoint_best.pth'), model, optimizer, scheduler, epoch, best_metric, args, split_cfg)
        latest_path = os.path.join(args.out_dir, 'checkpoint_latest.pth')
        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_metric, args, split_cfg)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(os.path.join(args.out_dir, f'checkpoint_epoch_{epoch:03d}.pth'), model, optimizer, scheduler, epoch, best_metric, args, split_cfg)


if __name__ == '__main__':
    main()
