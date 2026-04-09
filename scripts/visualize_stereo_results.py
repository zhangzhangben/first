import argparse
import csv
import os
import sys

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import yaml
from omegaconf import OmegaConf

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

from Utils import set_logging_format
from scripts.eval_booster import (
    collect_scene_samples,
    downsample_map_linear,
    downsample_mask_nearest,
    infer_single_pair,
    load_model_and_cfg,
    read_rgb,
    resolve_split_root,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--model_dir", default=None, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--setups", nargs="+", default=None)
    parser.add_argument("--scene_names", nargs="+", default=None)
    parser.add_argument("--max_samples_per_scene", default=2, type=int)
    return parser.parse_args()


def load_protocol(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def colorize_disparity(disp, maxval=None):
    disp = np.asarray(disp, dtype=np.float32)
    invalid = ~np.isfinite(disp) | (disp <= 0)
    valid = ~invalid
    if maxval is None:
        maxval = float(np.max(disp[valid])) if valid.any() else 1.0
    maxval = max(maxval, 1e-6)
    vis = np.clip(disp / maxval, 0.0, 1.0)
    vis = cv2.applyColorMap((vis * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[invalid] = 0
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def colorize_error(pred, gt, valid_mask):
    err = np.abs(pred - gt)
    vis = np.zeros((*err.shape, 3), dtype=np.uint8)
    bins = [
        (0.0, 1.0, (224, 243, 248)),
        (1.0, 2.0, (254, 224, 144)),
        (2.0, 4.0, (253, 174, 97)),
        (4.0, 8.0, (244, 109, 67)),
        (8.0, 1e9, (165, 0, 38)),
    ]
    for lo, hi, color in bins:
        mask = valid_mask & (err >= lo) & (err < hi)
        vis[mask] = np.asarray(color, dtype=np.uint8)
    return vis


def metric_resolution_arrays(left_rgb, gt_disp, pred_disp, setup, pred_meta, benchmark_gt_resolution):
    valid_mask = np.isfinite(gt_disp) & (gt_disp > 0.0)
    if benchmark_gt_resolution == "quarter":
        gt_metric = downsample_map_linear(gt_disp, 0.25, value_scale=0.25)
        valid_metric = downsample_mask_nearest(valid_mask, 0.25) > 0
        left_metric = cv2.resize(left_rgb, (gt_metric.shape[1], gt_metric.shape[0]), interpolation=cv2.INTER_LINEAR)
        if setup == "balanced" and abs(float(pred_meta["total_scale"]) - 0.25) < 1e-6:
            pred_metric = pred_meta["native_disp"]
        else:
            pred_metric = downsample_map_linear(pred_disp, 0.25, value_scale=0.25)
        return left_metric, gt_metric, pred_metric, valid_metric
    return left_rgb, gt_disp, pred_disp, valid_mask


def save_panel(left_rgb, gt_disp, pred_disp, valid_mask, save_path, title):
    max_disp = float(np.max(gt_disp[valid_mask])) if valid_mask.any() else max(float(np.max(pred_disp)), 1.0)
    gt_vis = colorize_disparity(np.where(valid_mask, gt_disp, 0.0), maxval=max_disp)
    pred_vis = colorize_disparity(np.where(valid_mask, pred_disp, 0.0), maxval=max_disp)
    err_vis = colorize_error(pred_disp, gt_disp, valid_mask)

    plt.figure(figsize=(16, 4))
    for idx, (name, img) in enumerate(
        [("Left", left_rgb), ("GT", gt_vis), ("Pred", pred_vis), ("Error", err_vis)],
        start=1,
    ):
        plt.subplot(1, 4, idx)
        plt.title(name)
        plt.imshow(img)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    protocol = load_protocol(args.config)
    data_cfg = protocol["data"]
    val_cfg = protocol["validation"]
    model_dir = args.model_dir or protocol["model"]["model_dir"]

    eval_args = OmegaConf.create(
        {
            "model_dir": model_dir,
            "booster_root": data_cfg["booster_root"],
            "split_dir_name": data_cfg["split_dir_name"],
            "setups": args.setups or data_cfg["val_setups"],
            "left_rel_dir": "camera_00",
            "left_pattern": "im*.png",
            "balanced_right_rel_dir": "camera_02",
            "unbalanced_right_rel_dir": "camera_01",
            "balanced_material_mask_name": "mask_cat.png",
            "unbalanced_material_mask_name": "warped_mask_cat.png",
            "left_occ_mask_name": "mask_00.png",
            "gt_disp_name": "disp_00.npy",
            "right_from_left_src": None,
            "right_from_left_dst": None,
            "out_dir": args.out_dir,
            "paper_protocol": val_cfg["paper_protocol"],
            "save_preds": 0,
            "pred_suffix": ".npy",
            "scale": data_cfg["scale"],
            "balanced_input_scale": data_cfg["balanced_input_scale"],
            "unbalanced_input_scale": data_cfg["unbalanced_input_scale"],
            "match_unbalanced_left_to_right": int(data_cfg["match_unbalanced_left_to_right"]),
            "benchmark_gt_resolution": val_cfg["benchmark_gt_resolution"],
            "hiera": int(val_cfg["hiera"]),
            "valid_iters": int(val_cfg["valid_iters"]),
            "max_disp": int(val_cfg["max_disp"]),
            "low_memory": int(val_cfg["low_memory"]),
            "class_ids": [0, 1, 2, 3],
            "scene_names": args.scene_names,
            "max_samples_per_scene": args.max_samples_per_scene,
        }
    )

    set_logging_format()
    model, resolved_args = load_model_and_cfg(model_dir, eval_args)
    split_root = resolve_split_root(data_cfg["booster_root"], data_cfg["split_dir_name"])
    scene_names = args.scene_names
    if scene_names is None:
        with open(data_cfg["split_file"], "r") as f:
            split_cfg = yaml.safe_load(f)
        scene_names = split_cfg["val_scenes"]

    manifest_rows = []
    for setup in resolved_args.setups:
        for scene in scene_names:
            scene_dir = os.path.join(split_root, setup, scene)
            samples = collect_scene_samples(scene_dir, setup, resolved_args)
            if args.max_samples_per_scene is not None:
                samples = samples[: args.max_samples_per_scene]
            for sample in samples:
                left_rgb = read_rgb(sample["left_file"])
                gt_disp = np.load(sample["gt_file"]).astype(np.float32)
                pred_disp, pred_meta = infer_single_pair(
                    model,
                    resolved_args,
                    setup,
                    sample["left_file"],
                    sample["right_file"],
                    return_native=True,
                )
                left_metric, gt_metric, pred_metric, valid_metric = metric_resolution_arrays(
                    left_rgb, gt_disp, pred_disp, setup, pred_meta, resolved_args.benchmark_gt_resolution
                )
                err = np.abs(pred_metric - gt_metric)[valid_metric]
                bad2 = float((err > 2.0).mean() * 100.0) if err.size else 0.0
                mae = float(err.mean()) if err.size else 0.0
                stem = f"{setup}_{scene}_{os.path.splitext(sample['left_name'])[0]}"
                save_path = os.path.join(args.out_dir, setup, scene, f"{stem}.png")
                title = f"{setup}/{scene}/{sample['left_name']} | Bad2={bad2:.2f} MAE={mae:.3f}"
                save_panel(left_metric, gt_metric, pred_metric, valid_metric, save_path, title)
                manifest_rows.append(
                    {
                        "setup": setup,
                        "scene": scene,
                        "left_name": sample["left_name"],
                        "bad2": bad2,
                        "mae": mae,
                        "panel_path": save_path,
                    }
                )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["setup", "scene", "left_name", "bad2", "mae", "panel_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
