import argparse
import json
import logging
import math
import os
import shlex
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

import Utils as U
from core.stereo_datasets import build_dataloader, build_dataset, load_scene_split
from core.utils.utils import InputPadder
from scripts.eval_booster import apply_protocol_defaults, downsample_map_linear, downsample_mask_nearest, infer_single_pair


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--amp_dtype", default=None, choices=["fp16", "bf16", "fp32"])
    return parser.parse_args()


def configure_amp_dtype(amp_dtype):
    if amp_dtype == "fp16":
        U.AMP_DTYPE = torch.float16
    elif amp_dtype == "bf16":
        U.AMP_DTYPE = torch.bfloat16
    elif amp_dtype == "fp32":
        U.AMP_DTYPE = torch.float32
    else:
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")


def merge_cli_overrides(protocol, cli_args):
    protocol = dict(protocol)
    protocol["outputs"] = dict(protocol["outputs"])
    protocol["train"] = dict(protocol["train"])
    if cli_args.out_dir is not None:
        protocol["outputs"]["out_dir"] = cli_args.out_dir
    if cli_args.resume is not None:
        protocol["train"]["resume"] = cli_args.resume
    if cli_args.epochs is not None:
        protocol["train"]["epochs"] = cli_args.epochs
    if cli_args.batch_size is not None:
        protocol["train"]["batch_size"] = cli_args.batch_size
    if cli_args.seed is not None:
        protocol["train"]["seed"] = cli_args.seed
    if cli_args.amp_dtype is not None:
        protocol["train"]["amp_dtype"] = cli_args.amp_dtype
    return protocol


def to_eval_cfg(protocol):
    model_cfg = protocol["model"]
    data_cfg = protocol["data"]
    val_cfg = protocol["validation"]
    train_cfg = protocol["train"]
    if data_cfg["dataset"].lower() == "booster":
        protocol_cfg = apply_protocol_defaults(
            {
                "paper_protocol": val_cfg["paper_protocol"],
                "scale": data_cfg["scale"],
                "balanced_input_scale": data_cfg["balanced_input_scale"],
                "unbalanced_input_scale": data_cfg["unbalanced_input_scale"],
                "benchmark_gt_resolution": val_cfg["benchmark_gt_resolution"],
                "low_memory": val_cfg["low_memory"],
                "max_disp": val_cfg["max_disp"],
            }
        )
    else:
        protocol_cfg = {
            "low_memory": val_cfg["low_memory"],
            "max_disp": val_cfg["max_disp"],
        }
    cfg = {
        "corr_levels": 2,
        "corr_radius": 4,
        "hidden_dims": [128],
        "low_memory": int(protocol_cfg["low_memory"]),
        "max_disp": int(protocol_cfg["max_disp"]),
        "mixed_precision": bool(train_cfg["amp_dtype"] != "fp32"),
        "n_downsample": 2,
        "n_gru_layers": 1,
        "slow_fast_gru": False,
        "valid_iters": int(val_cfg["valid_iters"]),
        "vit_size": "vitl",
    }
    if os.path.isfile(model_cfg["cfg_path"]):
        with open(model_cfg["cfg_path"], "r") as f:
            loaded = yaml.safe_load(f) or {}
        loaded.update(cfg)
        cfg = loaded
    return cfg


def unpad_scaled_tensor(x, padder, ref_hw):
    ref_h, ref_w = ref_hw
    scale_h = x.shape[-2] / float(ref_h)
    scale_w = x.shape[-1] / float(ref_w)
    pad_left = int(round(padder._pad[0] * scale_w))
    pad_right = int(round(padder._pad[1] * scale_w))
    pad_top = int(round(padder._pad[2] * scale_h))
    pad_bottom = int(round(padder._pad[3] * scale_h))
    ht, wd = x.shape[-2:]
    return x[..., pad_top:ht - pad_bottom, pad_left:wd - pad_right]


def downsample_disp_gt_for_fast_init(disp_gt, valid, target_hw):
    scale_w = target_hw[1] / float(disp_gt.shape[-1])
    gt_small = F.interpolate(disp_gt, size=target_hw, mode="bilinear", align_corners=True) * scale_w
    valid_small = F.interpolate(valid.float(), size=target_hw, mode="nearest")
    return gt_small, valid_small


def fast_sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192, init_weight=1.0):
    n_predictions = len(disp_preds)
    if n_predictions < 1:
        raise ValueError("Expected at least one disparity prediction")

    mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
    valid_mask = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    if valid_mask.sum() == 0:
        zero = disp_gt.new_tensor(0.0)
        return zero, {"epe": 0.0, "1px": 0.0, "3px": 0.0, "5px": 0.0}

    init_gt, init_valid = downsample_disp_gt_for_fast_init(disp_gt, valid.unsqueeze(1), disp_init_pred.shape[-2:])
    init_mag = torch.sum(init_gt ** 2, dim=1).sqrt()
    init_valid_mask = ((init_valid[:, 0] >= 0.5) & (init_mag < (float(max_disp) * 0.25))).unsqueeze(1)
    if init_valid_mask.sum() > 0:
        disp_loss = float(init_weight) * F.smooth_l1_loss(
            disp_init_pred[init_valid_mask], init_gt[init_valid_mask], reduction="mean"
        )
    else:
        disp_loss = disp_gt.new_tensor(0.0)

    adjusted_loss_gamma = 1.0 if n_predictions == 1 else float(loss_gamma) ** (15.0 / float(n_predictions - 1))
    for i, pred in enumerate(disp_preds):
        i_weight = adjusted_loss_gamma ** float(n_predictions - i - 1)
        i_loss = (pred - disp_gt).abs()
        disp_loss = disp_loss + float(i_weight) * i_loss[valid_mask].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid_mask.view(-1)]
    metrics = {
        "epe": float(epe.mean().item()),
        "1px": float((epe < 1).float().mean().item()),
        "3px": float((epe < 3).float().mean().item()),
        "5px": float((epe < 5).float().mean().item()),
    }
    return disp_loss, metrics


def masked_l1(pred, gt, valid):
    valid = valid > 0.5
    if valid.sum() == 0:
        return pred.new_tensor(0.0)
    return (pred[valid] - gt[valid]).abs().mean()


def masked_epe(pred, gt, valid):
    valid = valid > 0.5
    if valid.sum() == 0:
        return pred.new_tensor(0.0)
    epe = torch.sqrt(torch.square(pred - gt) + 1e-12)
    return epe[valid].mean()


def compute_formal_booster_metrics(model, batch, protocol):
    data_cfg = protocol["data"]
    val_cfg = protocol["validation"]
    pred_disp, pred_meta = infer_single_pair(
        model,
        OmegaConf.create(
            {
                "balanced_input_scale": data_cfg["balanced_input_scale"],
                "unbalanced_input_scale": data_cfg["unbalanced_input_scale"],
                "scale": data_cfg["scale"],
                "match_unbalanced_left_to_right": int(data_cfg["match_unbalanced_left_to_right"]),
                "benchmark_gt_resolution": val_cfg["benchmark_gt_resolution"],
                "hiera": int(val_cfg["hiera"]),
                "valid_iters": int(val_cfg["valid_iters"]),
                "max_disp": int(val_cfg["max_disp"]),
                "low_memory": int(val_cfg["low_memory"]),
            }
        ),
        batch["setup"],
        batch["left_file"],
        batch["right_file"],
        return_native=True,
    )
    gt_disp = np.load(batch["gt_file"]).astype(np.float32)
    valid_mask = np.isfinite(gt_disp) & (gt_disp > 0.0)
    if val_cfg["benchmark_gt_resolution"] == "quarter":
        gt_metric = downsample_map_linear(gt_disp, 0.25, value_scale=0.25)
        valid_mask_metric = downsample_mask_nearest(valid_mask, 0.25) > 0
        if batch["setup"] == "balanced" and abs(float(pred_meta["total_scale"]) - 0.25) < 1e-6:
            pred_metric = pred_meta["native_disp"]
        else:
            pred_metric = downsample_map_linear(pred_disp, 0.25, value_scale=0.25)
    else:
        gt_metric = gt_disp
        pred_metric = pred_disp
        valid_mask_metric = valid_mask
    metric_errors = np.abs(pred_metric - gt_metric)[valid_mask_metric]
    return {
        "num_pixels": int(metric_errors.size),
        "bad2_count": int((metric_errors > 2.0).sum()),
        "mae_sum": float(metric_errors.sum()),
    }


def run_validation(model, loader, protocol):
    model.eval()
    dataset_name = protocol["data"]["dataset"].lower()
    totals = {"loss_sum": 0.0, "epe_sum": 0.0, "num_batches": 0, "num_pixels": 0, "bad2_count": 0, "mae_sum": 0.0}
    max_val_batches = protocol["train"]["max_val_batches"]
    val_cfg = protocol["validation"]
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            left = batch["left"].cuda(non_blocking=True).unsqueeze(0)
            right = batch["right"].cuda(non_blocking=True).unsqueeze(0)
            gt = batch["disp"].cuda(non_blocking=True).unsqueeze(0)
            valid = batch["valid"].cuda(non_blocking=True).unsqueeze(0)
            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left, right = padder.pad(left, right)
            with torch.amp.autocast("cuda", enabled=protocol["train"]["amp_dtype"] != "fp32", dtype=U.AMP_DTYPE):
                pred = model.forward(
                    left,
                    right,
                    iters=int(val_cfg["valid_iters"]),
                    test_mode=True,
                    low_memory=bool(val_cfg["low_memory"]),
                    optimize_build_volume="pytorch1",
                )
            pred = padder.unpad(pred.float())
            totals["loss_sum"] += float(masked_l1(pred, gt, valid).item())
            totals["epe_sum"] += float(masked_epe(pred, gt, valid).item())
            totals["num_batches"] += 1
            if dataset_name == "booster":
                formal = compute_formal_booster_metrics(model, batch, protocol)
                totals["num_pixels"] += formal["num_pixels"]
                totals["bad2_count"] += formal["bad2_count"]
                totals["mae_sum"] += formal["mae_sum"]
            if max_val_batches is not None and batch_idx >= int(max_val_batches):
                break
    results = {
        "loss": totals["loss_sum"] / max(totals["num_batches"], 1),
        "epe": totals["epe_sum"] / max(totals["num_batches"], 1),
    }
    if dataset_name == "booster":
        results["bad2"] = totals["bad2_count"] * 100.0 / max(totals["num_pixels"], 1)
        results["mae"] = totals["mae_sum"] / max(totals["num_pixels"], 1)
    else:
        results["bad2"] = None
        results["mae"] = None
    return results


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, protocol, split_cfg):
    ckpt = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "protocol": protocol,
        "split_cfg": split_cfg,
    }
    torch.save(ckpt, path)


def save_serialized_model(path, model):
    model_cpu = model.cpu()
    torch.save(model_cpu, path)
    model.cuda()


def save_eval_cfg(path, protocol):
    cfg = to_eval_cfg(protocol)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)


def save_command(path):
    with open(path, "w") as f:
        f.write(" ".join(shlex.quote(x) for x in sys.argv) + "\n")


def main():
    cli_args = parse_args()
    with open(cli_args.config, "r") as f:
        protocol = yaml.safe_load(f)
    protocol = merge_cli_overrides(protocol, cli_args)

    train_cfg = protocol["train"]
    data_cfg = protocol["data"]
    model_cfg = protocol["model"]
    out_dir = protocol["outputs"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    U.set_logging_format()
    U.set_seed(int(train_cfg["seed"]))
    configure_amp_dtype(train_cfg["amp_dtype"])

    with open(os.path.join(out_dir, "protocol_resolved.yaml"), "w") as f:
        yaml.safe_dump(protocol, f, sort_keys=False)
    save_command(os.path.join(out_dir, "command.sh"))
    if data_cfg["dataset"].lower() == "booster":
        split_cfg = load_scene_split(data_cfg["split_file"])
        with open(os.path.join(out_dir, "split_used.yaml"), "w") as f:
            yaml.safe_dump(split_cfg, f, sort_keys=False)
    else:
        split_cfg = {}

    model = torch.load(model_cfg["model_dir"], map_location="cpu", weights_only=False)
    eval_cfg = to_eval_cfg(protocol)
    model.args.valid_iters = int(protocol["validation"]["valid_iters"])
    model.args.max_disp = int(protocol["validation"]["max_disp"])
    model.args.low_memory = int(protocol["validation"]["low_memory"])
    model.args.mixed_precision = bool(train_cfg["amp_dtype"] != "fp32")
    model.args.corr_levels = int(eval_cfg["corr_levels"])
    model.args.corr_radius = int(eval_cfg["corr_radius"])
    model.cuda()

    train_dataset = build_dataset(dict(data_cfg), subset="train")
    val_dataset = build_dataset(dict(data_cfg), subset="val")
    train_loader = build_dataloader(
        train_dataset, batch_size=int(train_cfg["batch_size"]), num_workers=int(train_cfg["num_workers"]), shuffle=True
    )
    val_loader = build_dataloader(val_dataset, batch_size=1, num_workers=int(train_cfg["num_workers"]), shuffle=False)

    optimizer_cfg = protocol["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
        eps=float(optimizer_cfg["eps"]),
    )

    total_steps = len(train_loader) * int(train_cfg["epochs"])
    scheduler_cfg = protocol["scheduler"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(optimizer_cfg["lr"]),
        total_steps=max(total_steps + int(scheduler_cfg["extra_steps"]), 1),
        pct_start=float(scheduler_cfg["pct_start"]),
        cycle_momentum=bool(scheduler_cfg["cycle_momentum"]),
        anneal_strategy=str(scheduler_cfg["anneal_strategy"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg["amp_dtype"] == "fp16"))

    history = []
    best_metric = math.inf
    start_epoch = 0
    if train_cfg.get("resume"):
        ckpt = torch.load(train_cfg["resume"], map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        best_metric = float(ckpt["best_metric"])
        start_epoch = int(ckpt["epoch"]) + 1

    logging.info(f"model_dir: {model_cfg['model_dir']}")
    logging.info(f"dataset: {data_cfg['dataset']}")
    logging.info(f"train samples: {len(train_dataset)}")
    logging.info(f"val samples: {len(val_dataset)}")
    logging.info(f"out_dir: {out_dir}")
    logging.info(f"loss: {protocol['loss']['name']} ({protocol['loss']['source']})")
    logging.info(f"optimizer: {protocol['optimizer']['name']}")
    logging.info(f"scheduler: {protocol['scheduler']['name']}")

    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        model.train()
        epoch_loss = 0.0
        epoch_epe = 0.0
        epoch_steps = 0
        for step, batch in enumerate(train_loader, start=1):
            left = batch["left"].cuda(non_blocking=True).unsqueeze(0)
            right = batch["right"].cuda(non_blocking=True).unsqueeze(0)
            gt = batch["disp"].cuda(non_blocking=True).unsqueeze(0)
            valid = batch["valid"].cuda(non_blocking=True).unsqueeze(0)
            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left, right = padder.pad(left, right)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=train_cfg["amp_dtype"] != "fp32", dtype=U.AMP_DTYPE):
                init_disp, preds = model.forward(
                    left,
                    right,
                    iters=int(train_cfg["train_iters"]),
                    test_mode=False,
                    low_memory=bool(protocol["validation"]["low_memory"]),
                    optimize_build_volume="pytorch1",
                )
                init_disp = unpad_scaled_tensor(init_disp, padder, left.shape[-2:])
                preds = [padder.unpad(pred) for pred in preds]
                loss, metrics = fast_sequence_loss(
                    preds,
                    init_disp,
                    gt,
                    valid[:, 0],
                    loss_gamma=float(train_cfg["sequence_loss_gamma"]),
                    max_disp=int(protocol["validation"]["max_disp"]),
                    init_weight=float(protocol["loss"]["init_weight"]),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            epoch_loss += float(loss.item())
            epoch_epe += float(metrics["epe"])
            epoch_steps += 1
            if step % int(protocol["logging"]["log_every"]) == 0:
                logging.info(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.6f} epe={metrics['epe']:.6f} "
                    f"1px={metrics['1px']:.4f} 3px={metrics['3px']:.4f}"
                )
            if train_cfg["max_train_batches"] is not None and step >= int(train_cfg["max_train_batches"]):
                break

        val_metrics = run_validation(model, val_loader, protocol)
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(epoch_steps, 1),
            "train_epe": epoch_epe / max(epoch_steps, 1),
            "val_loss": val_metrics["loss"],
            "val_epe_monitor": val_metrics["epe"],
            "val_bad2_formal": val_metrics["bad2"],
            "val_mae_formal": val_metrics["mae"],
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)
        if record["val_bad2_formal"] is None:
            logging.info(
                f"epoch={epoch} train_loss={record['train_loss']:.6f} train_epe={record['train_epe']:.6f} "
                f"val_loss={record['val_loss']:.6f} val_epe_monitor={record['val_epe_monitor']:.6f}"
            )
        else:
            logging.info(
                f"epoch={epoch} train_loss={record['train_loss']:.6f} train_epe={record['train_epe']:.6f} "
                f"val_loss={record['val_loss']:.6f} val_epe_monitor={record['val_epe_monitor']:.6f} "
                f"val_bad2_formal={record['val_bad2_formal']:.4f} val_mae_formal={record['val_mae_formal']:.6f}"
            )
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if (epoch + 1) % int(protocol["logging"]["save_every"]) == 0:
            save_checkpoint(os.path.join(out_dir, f"checkpoint_epoch_{epoch:03d}.pth"), model, optimizer, scheduler, epoch, best_metric, protocol, split_cfg)
        save_checkpoint(os.path.join(out_dir, "checkpoint_latest.pth"), model, optimizer, scheduler, epoch, best_metric, protocol, split_cfg)
        if protocol["logging"]["save_serialized_model"]:
            save_serialized_model(os.path.join(out_dir, "model_latest_serialize.pth"), model)
            if protocol["logging"]["export_cfg_for_eval"]:
                save_eval_cfg(os.path.join(out_dir, "cfg.yaml"), protocol)

        monitor_metric = record["val_bad2_formal"] if record["val_bad2_formal"] is not None else record["val_epe_monitor"]
        if monitor_metric < best_metric:
            best_metric = monitor_metric
            save_checkpoint(os.path.join(out_dir, "checkpoint_best.pth"), model, optimizer, scheduler, epoch, best_metric, protocol, split_cfg)
            if protocol["logging"]["save_serialized_model"]:
                save_serialized_model(os.path.join(out_dir, "model_best_serialize.pth"), model)
                if protocol["logging"]["export_cfg_for_eval"]:
                    save_eval_cfg(os.path.join(out_dir, "cfg.yaml"), protocol)


if __name__ == "__main__":
    main()
