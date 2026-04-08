import argparse
import csv
import json
import logging
import os
import sys
from glob import glob

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

from Utils import AMP_DTYPE, set_logging_format, set_seed
from core.utils.utils import InputPadder


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--model_dir",
      default=f"{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth",
      type=str,
  )
  parser.add_argument("--booster_root", type=str, required=True)
  parser.add_argument("--split_dir_name", default="train", type=str)
  parser.add_argument("--setups", nargs="+", default=["balanced", "unbalanced"])
  parser.add_argument("--left_rel_dir", default="camera_00", type=str)
  parser.add_argument("--left_pattern", default="im*.png", type=str)
  parser.add_argument("--balanced_right_rel_dir", default="camera_02", type=str)
  parser.add_argument("--unbalanced_right_rel_dir", default="camera_01", type=str)
  parser.add_argument("--balanced_material_mask_name", default="mask_cat.png", type=str)
  parser.add_argument("--unbalanced_material_mask_name", default="warped_mask_cat.png", type=str)
  parser.add_argument("--left_occ_mask_name", default="mask_00.png", type=str)
  parser.add_argument("--gt_disp_name", default="disp_00.npy", type=str)
  parser.add_argument("--right_from_left_src", default=None, type=str)
  parser.add_argument("--right_from_left_dst", default=None, type=str)
  parser.add_argument("--out_dir", default=f"{code_dir}/../output_booster_eval", type=str)
  parser.add_argument("--paper_protocol", default="booster_q", choices=["legacy", "booster_full", "booster_q"])
  parser.add_argument("--save_preds", default=0, type=int)
  parser.add_argument("--pred_suffix", default=".npy", type=str)
  parser.add_argument("--scale", default=None, type=float)
  parser.add_argument("--balanced_input_scale", default=None, type=float)
  parser.add_argument("--unbalanced_input_scale", default=None, type=float)
  parser.add_argument("--match_unbalanced_left_to_right", default=1, type=int)
  parser.add_argument("--benchmark_gt_resolution", default=None, choices=["full", "quarter"])
  parser.add_argument("--hiera", default=0, type=int)
  parser.add_argument("--valid_iters", default=None, type=int)
  parser.add_argument("--max_disp", default=None, type=int)
  parser.add_argument("--low_memory", default=None, type=int)
  parser.add_argument('--use_uncertainty_update_gate', default=None, type=int)
  parser.add_argument('--uncertainty_gate_scale', default=None, type=float)
  parser.add_argument('--uncertainty_gate_bias', default=None, type=float)
  parser.add_argument('--uncertainty_gate_hidden_dim', default=None, type=int)
  parser.add_argument('--use_loslite_refinement', default=None, type=int)
  parser.add_argument('--loslite_hidden_dim', default=None, type=int)
  parser.add_argument('--loslite_uncertainty_margin', default=None, type=float)
  parser.add_argument('--loslite_propagation_blend', default=None, type=float)
  parser.add_argument('--loslite_grad_scale', default=None, type=float)
  parser.add_argument('--loslite_offset_scale', default=None, type=float)
  parser.add_argument("--class_ids", nargs="+", default=[0, 1, 2, 3], type=int)
  parser.add_argument("--scene_names", nargs="+", default=None)
  parser.add_argument("--max_samples_per_scene", default=None, type=int)
  return parser.parse_args()


def resolve_split_root(booster_root, split_dir_name):
  candidates = [
      os.path.join(booster_root, split_dir_name),
      os.path.join(booster_root, split_dir_name.capitalize()),
      booster_root,
  ]
  for candidate in candidates:
    if os.path.isdir(candidate):
      return candidate
  raise FileNotFoundError(f"Cannot find BOOSTER split root under {booster_root}")


def get_right_rel_dir(args, setup):
  if setup == "balanced":
    return args.balanced_right_rel_dir
  if setup == "unbalanced":
    return args.unbalanced_right_rel_dir
  raise ValueError(f"Unknown setup: {setup}")


def get_material_mask_name(args, setup):
  if setup == "balanced":
    return args.balanced_material_mask_name
  if setup == "unbalanced":
    return args.unbalanced_material_mask_name
  raise ValueError(f"Unknown setup: {setup}")


def collect_scene_samples(scene_dir, setup, args):
  left_dir = os.path.join(scene_dir, args.left_rel_dir)
  right_dir = os.path.join(scene_dir, get_right_rel_dir(args, setup))
  gt_file = os.path.join(scene_dir, args.gt_disp_name)
  occ_mask_file = os.path.join(scene_dir, args.left_occ_mask_name)
  material_mask_file = os.path.join(scene_dir, get_material_mask_name(args, setup))

  if not os.path.isfile(gt_file):
    raise FileNotFoundError(f"GT disparity not found: {gt_file}")

  left_files = sorted(glob(os.path.join(left_dir, args.left_pattern)))
  samples = []
  for left_file in left_files:
    left_name = os.path.basename(left_file)
    if args.right_from_left_src is not None and args.right_from_left_dst is not None:
      if args.right_from_left_src not in left_name:
        raise ValueError(f"Cannot map right image for {left_file}: missing token {args.right_from_left_src}")
      right_name = left_name.replace(args.right_from_left_src, args.right_from_left_dst, 1)
    else:
      right_name = left_name
    right_file = os.path.join(right_dir, right_name)
    if not os.path.isfile(right_file):
      raise FileNotFoundError(f"Right image not found for {left_file}: {right_file}")
    samples.append({
        "left_file": left_file,
        "right_file": right_file,
        "left_name": left_name,
        "gt_file": gt_file,
        "occ_mask_file": occ_mask_file if os.path.isfile(occ_mask_file) else None,
        "material_mask_file": material_mask_file if os.path.isfile(material_mask_file) else None,
    })
  return samples


def read_rgb(path):
  img = imageio.imread(path)
  if img.ndim == 2:
    img = np.tile(img[..., None], (1, 1, 3))
  return img[..., :3]


def apply_protocol_defaults(cfg):
  protocol = cfg.get("paper_protocol", "legacy")
  if cfg.get("scale") is None:
    cfg["scale"] = 1.0
  if cfg.get("balanced_input_scale") is None:
    cfg["balanced_input_scale"] = 0.25 if protocol == "booster_q" else 1.0
  if cfg.get("unbalanced_input_scale") is None:
    cfg["unbalanced_input_scale"] = 1.0
  if cfg.get("benchmark_gt_resolution") is None:
    cfg["benchmark_gt_resolution"] = "quarter" if protocol == "booster_q" else "full"
  if cfg.get("low_memory") is None:
    cfg["low_memory"] = 1 if protocol == "booster_q" else 0
  if cfg.get("max_disp") is None and protocol == "booster_q":
    cfg["max_disp"] = 192
  return cfg


def load_serialized_cfg(model_dir, model=None):
  cfg_path = os.path.join(os.path.dirname(model_dir), "cfg.yaml")
  if os.path.isfile(cfg_path):
    with open(cfg_path, "r") as f:
      return yaml.safe_load(f) or {}

  if model is not None and hasattr(model, "args") and model.args is not None:
    if OmegaConf.is_config(model.args):
      return OmegaConf.to_container(model.args, resolve=True)
    if isinstance(model.args, dict):
      return dict(model.args)
    return {
        k: v for k, v in vars(model.args).items()
        if not k.startswith("_")
    }

  return {}


def load_model_and_cfg(model_dir, cli_args):
  loaded = torch.load(model_dir, map_location="cpu", weights_only=False)
  if isinstance(loaded, dict):
    raise RuntimeError(
        f"{model_dir} looks like a checkpoint dict. Please use a serialized model such as model_best_bp2_serialize.pth"
    )

  model = loaded
  model.cuda().eval()

  if hasattr(model, "args") and model.args is not None:
    if cli_args.valid_iters is not None:
      model.args.valid_iters = cli_args.valid_iters
    if cli_args.max_disp is not None:
      model.args.max_disp = cli_args.max_disp
    if cli_args.low_memory is not None:
      model.args.low_memory = cli_args.low_memory
    if cli_args.use_uncertainty_update_gate is not None:
      model.args.use_uncertainty_update_gate = bool(cli_args.use_uncertainty_update_gate)
    if cli_args.uncertainty_gate_scale is not None:
      model.args.uncertainty_gate_scale = cli_args.uncertainty_gate_scale
    if cli_args.uncertainty_gate_bias is not None:
      model.args.uncertainty_gate_bias = cli_args.uncertainty_gate_bias
    if cli_args.uncertainty_gate_hidden_dim is not None:
      model.args.uncertainty_gate_hidden_dim = cli_args.uncertainty_gate_hidden_dim
    if cli_args.use_loslite_refinement is not None:
      model.args.use_loslite_refinement = bool(cli_args.use_loslite_refinement)
    if cli_args.loslite_hidden_dim is not None:
      model.args.loslite_hidden_dim = cli_args.loslite_hidden_dim
    if cli_args.loslite_uncertainty_margin is not None:
      model.args.loslite_uncertainty_margin = cli_args.loslite_uncertainty_margin
    if cli_args.loslite_propagation_blend is not None:
      model.args.loslite_propagation_blend = cli_args.loslite_propagation_blend
    if cli_args.loslite_grad_scale is not None:
      model.args.loslite_grad_scale = cli_args.loslite_grad_scale
    if cli_args.loslite_offset_scale is not None:
      model.args.loslite_offset_scale = cli_args.loslite_offset_scale
  else:
    cfg = load_serialized_cfg(model_dir, model)
    cfg.setdefault("normalize", True)
    cfg.setdefault("corr_levels", 2)
    cfg.setdefault("corr_radius", 4)
    cfg.setdefault("mixed_precision", True)
    cfg.setdefault("low_memory", 0)
    cfg.setdefault("n_gru_layers", 1)
    cfg.setdefault("slow_fast_gru", False)
    if cli_args.valid_iters is not None:
      cfg["valid_iters"] = cli_args.valid_iters
    if cli_args.max_disp is not None:
      cfg["max_disp"] = cli_args.max_disp
    model.args = OmegaConf.create(cfg)

  cfg = load_serialized_cfg(model_dir, model)
  for k, v in vars(cli_args).items():
    if k not in cfg or v is not None:
      cfg[k] = v
  if hasattr(model, "args") and model.args is not None:
    cfg["valid_iters"] = int(model.args.valid_iters)
    cfg["max_disp"] = int(model.args.max_disp)
  cfg = apply_protocol_defaults(cfg)
  return model, OmegaConf.create(cfg)


def resize_pair(img0, img1, scale_factor):
  if scale_factor == 1.0:
    return img0, img1
  new_w = max(1, int(round(img0.shape[1] * scale_factor)))
  new_h = max(1, int(round(img0.shape[0] * scale_factor)))
  img0 = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
  img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
  return img0, img1


def infer_single_pair(model, args, setup, left_file, right_file, return_native=False):
  img0 = read_rgb(left_file)
  img1 = read_rgb(right_file)
  ori_h, ori_w = img0.shape[:2]
  disp_scale_x = 1.0

  if setup == "unbalanced" and args.match_unbalanced_left_to_right and img1.shape[:2] != img0.shape[:2]:
    target_h, target_w = img1.shape[:2]
    img0 = cv2.resize(img0, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    disp_scale_x *= target_w / float(ori_w)
  elif img1.shape[:2] != img0.shape[:2]:
    raise ValueError(
        f"Unexpected left/right size mismatch for setup={setup}: left={img0.shape[:2]} right={img1.shape[:2]}"
    )

  setup_scale = args.balanced_input_scale if setup == "balanced" else args.unbalanced_input_scale
  total_scale = float(setup_scale) * float(args.scale)
  img0, img1 = resize_pair(img0, img1, total_scale)
  disp_scale_x *= total_scale

  infer_h, infer_w = img0.shape[:2]
  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
  padder = InputPadder(img0.shape, divis_by=32, force_square=False)
  img0, img1 = padder.pad(img0, img1)

  with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
    if not args.hiera:
      disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True, low_memory=bool(args.low_memory), optimize_build_volume="pytorch1")
    else:
      disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, low_memory=bool(args.low_memory), small_ratio=0.5)

  disp = padder.unpad(disp.float())
  native_disp = disp.data.cpu().numpy().reshape(infer_h, infer_w).clip(0, None)
  disp = native_disp

  if disp_scale_x != 1.0 or infer_h != ori_h or infer_w != ori_w:
    disp = cv2.resize(disp, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
    disp = disp / disp_scale_x

  if return_native:
    return disp, {
        "native_disp": native_disp,
        "native_shape": (infer_h, infer_w),
        "disp_scale_x": disp_scale_x,
        "ori_shape": (ori_h, ori_w),
        "total_scale": total_scale,
    }
  return disp


def read_mask_png(mask_path):
  mask = imageio.imread(mask_path)
  if mask.ndim == 3:
    mask = mask[..., 0]
  return mask


def downsample_map_linear(arr, scale, value_scale=1.0):
  h, w = arr.shape[:2]
  new_w = max(1, int(round(w * scale)))
  new_h = max(1, int(round(h * scale)))
  out = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
  if value_scale != 1.0:
    out = out * value_scale
  return out


def downsample_mask_nearest(mask, scale):
  h, w = mask.shape[:2]
  new_w = max(1, int(round(w * scale)))
  new_h = max(1, int(round(h * scale)))
  return cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def summarize_errors(errors):
  return {
      'bad_2': float((errors > 2.0).mean() * 100.0),
      'bad_4': float((errors > 4.0).mean() * 100.0),
      'bad_6': float((errors > 6.0).mean() * 100.0),
      'bad_8': float((errors > 8.0).mean() * 100.0),
      'mae': float(errors.mean()),
      'rmse': float(np.sqrt(np.square(errors).mean())),
  }


def append_benchmark_row(rows, setup, split_name, occlusion, gt_resolution, cls_name, errors):
  row = {
      'Setup': setup.capitalize(),
      'Split': split_name.capitalize(),
      'Occlusion': occlusion,
      'GT Resolution': gt_resolution,
      'Class': cls_name,
      'Num Pixels': int(errors.size),
  }
  row.update({
      'Bad 2': '',
      'Bad 4': '',
      'Bad 6': '',
      'Bad 8': '',
      'MAE': '',
      'RMSE': '',
  })
  if errors.size > 0:
    metrics = summarize_errors(errors)
    row.update({
        'Bad 2': metrics['bad_2'],
        'Bad 4': metrics['bad_4'],
        'Bad 6': metrics['bad_6'],
        'Bad 8': metrics['bad_8'],
        'MAE': metrics['mae'],
        'RMSE': metrics['rmse'],
    })
  rows.append(row)


def aggregate_benchmark_rows(rows):
  aggregates = {}
  for row in rows:
    key = (row["Setup"], row["Split"], row["Occlusion"], row["GT Resolution"], row["Class"])
    num_pixels = int(row["Num Pixels"])
    item = aggregates.setdefault(
        key,
        {
            "Num Pixels": 0,
            "sum_abs_err": 0.0,
            "sum_sq_err": 0.0,
            "bad_2_count": 0.0,
            "bad_4_count": 0.0,
            "bad_6_count": 0.0,
            "bad_8_count": 0.0,
        },
    )
    if num_pixels == 0:
      continue
    item["Num Pixels"] += num_pixels
    item["sum_abs_err"] += float(row["MAE"]) * num_pixels
    item["sum_sq_err"] += (float(row["RMSE"]) ** 2) * num_pixels
    item["bad_2_count"] += float(row["Bad 2"]) * num_pixels / 100.0
    item["bad_4_count"] += float(row["Bad 4"]) * num_pixels / 100.0
    item["bad_6_count"] += float(row["Bad 6"]) * num_pixels / 100.0
    item["bad_8_count"] += float(row["Bad 8"]) * num_pixels / 100.0

  summary_rows = []
  for key in sorted(aggregates.keys()):
    item = aggregates[key]
    num_pixels = item["Num Pixels"]
    row = {
        "Setup": key[0],
        "Split": key[1],
        "Occlusion": key[2],
        "GT Resolution": key[3],
        "Class": key[4],
        "Num Pixels": num_pixels,
        "Bad 2": "",
        "Bad 4": "",
        "Bad 6": "",
        "Bad 8": "",
        "MAE": "",
        "RMSE": "",
    }
    if num_pixels > 0:
      row.update({
          "Bad 2": item["bad_2_count"] * 100.0 / num_pixels,
          "Bad 4": item["bad_4_count"] * 100.0 / num_pixels,
          "Bad 6": item["bad_6_count"] * 100.0 / num_pixels,
          "Bad 8": item["bad_8_count"] * 100.0 / num_pixels,
          "MAE": item["sum_abs_err"] / num_pixels,
          "RMSE": float(np.sqrt(item["sum_sq_err"] / num_pixels)),
      })
    summary_rows.append(row)
  return summary_rows


def validate_scene_annotations(gt_disp, occ_mask, material_mask, setup, scene_name):
  if gt_disp.ndim != 2:
    raise ValueError(f"GT disparity must be HxW, got {gt_disp.shape} for {setup}/{scene_name}")
  if occ_mask is not None and occ_mask.shape != gt_disp.shape:
    raise ValueError(
        f"Occlusion mask shape {occ_mask.shape} does not match GT shape {gt_disp.shape} for {setup}/{scene_name}"
    )
  if material_mask is not None and material_mask.shape != gt_disp.shape:
    raise ValueError(
        f"Material mask shape {material_mask.shape} does not match GT shape {gt_disp.shape} for {setup}/{scene_name}"
    )


def build_track_definitions(class_ids):
  tracks = [
      {"name": "all", "kind": "all", "class_id": None},
      {"name": "noc", "kind": "noc", "class_id": None},
  ]
  for class_id in class_ids:
    tracks.append({"name": f"class_{class_id}", "kind": "class", "class_id": int(class_id)})
    tracks.append({"name": f"noc_class_{class_id}", "kind": "noc_class", "class_id": int(class_id)})
  return tracks


def init_aggregates(track_names):
  stats = {}
  for track_name in track_names:
    stats[track_name] = {
        "num_pixels": 0,
        "sum_abs_err": 0.0,
        "sum_sq_err": 0.0,
        "bad_2_count": 0,
        "bad_4_count": 0,
        "bad_6_count": 0,
        "bad_8_count": 0,
        "num_images": 0,
    }
  return stats


def update_stats(store, track_name, errors):
  if errors.size == 0:
    return
  item = store[track_name]
  item["num_pixels"] += int(errors.size)
  item["sum_abs_err"] += float(errors.sum())
  item["sum_sq_err"] += float(np.square(errors).sum())
  item["bad_2_count"] += int((errors > 2.0).sum())
  item["bad_4_count"] += int((errors > 4.0).sum())
  item["bad_6_count"] += int((errors > 6.0).sum())
  item["bad_8_count"] += int((errors > 8.0).sum())
  item["num_images"] += 1


def finalize_stats(store):
  out = {}
  for track_name, item in store.items():
    num_pixels = item["num_pixels"]
    if num_pixels == 0:
      out[track_name] = {
          "available": False,
          "num_images": item["num_images"],
          "num_pixels": 0,
      }
      continue
    out[track_name] = {
        "available": True,
        "num_images": item["num_images"],
        "num_pixels": num_pixels,
        "bad_2": item["bad_2_count"] * 100.0 / num_pixels,
        "bad_4": item["bad_4_count"] * 100.0 / num_pixels,
        "bad_6": item["bad_6_count"] * 100.0 / num_pixels,
        "bad_8": item["bad_8_count"] * 100.0 / num_pixels,
        "mae": item["sum_abs_err"] / num_pixels,
        "rmse": float(np.sqrt(item["sum_sq_err"] / num_pixels)),
    }
  return out


def compute_track_masks(valid_mask, occ_mask, material_mask, track_defs):
  masks = {}
  for track in track_defs:
    track_mask = valid_mask.copy()
    if track["kind"] in ("noc", "noc_class"):
      if occ_mask is None:
        masks[track["name"]] = None
        continue
      track_mask = track_mask & occ_mask
    if track["kind"] in ("class", "noc_class"):
      if material_mask is None:
        masks[track["name"]] = None
        continue
      track_mask = track_mask & (material_mask == track["class_id"])
    masks[track["name"]] = track_mask
  return masks


def save_pred_if_needed(pred_disp, sample, setup, scene_name, args):
  if not args.save_preds:
    return None
  stem = os.path.splitext(sample["left_name"])[0]
  out_dir = os.path.join(args.out_dir, "predictions", setup, scene_name)
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, stem + args.pred_suffix)
  if args.pred_suffix == ".npy":
    np.save(out_path, pred_disp.astype(np.float32))
  else:
    raise ValueError(f"Unsupported pred_suffix: {args.pred_suffix}")
  return out_path


def log_track_summary(prefix, stats, track_names):
  for track_name in track_names:
    track = stats.get(track_name, {})
    if not track.get("available"):
      continue
    logging.info(
        f"[{prefix}/{track_name}] pixels={track['num_pixels']} "
        f"Bad2={track['bad_2']:.4f} Bad4={track['bad_4']:.4f} "
        f"Bad6={track['bad_6']:.4f} Bad8={track['bad_8']:.4f} "
        f"MAE={track['mae']:.4f} RMSE={track['rmse']:.4f}"
    )


def main():
  cli_args = parse_args()
  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)

  model, args = load_model_and_cfg(cli_args.model_dir, cli_args)
  split_root = resolve_split_root(args.booster_root, args.split_dir_name)
  os.makedirs(args.out_dir, exist_ok=True)

  track_defs = build_track_definitions(args.class_ids)
  track_names = [track["name"] for track in track_defs]
  global_stats = init_aggregates(track_names)
  setup_stats = {setup: init_aggregates(track_names) for setup in args.setups}
  sample_rows = []
  scene_summary = {}
  benchmark_quarter_rows = []

  logging.info(f"booster_root: {args.booster_root}")
  logging.info(f"split_root: {split_root}")
  logging.info(f"model_dir: {args.model_dir}")
  logging.info(f"setups: {list(args.setups)}")
  logging.info(f"split_dir_name: {args.split_dir_name}")
  logging.info(f"left_rel_dir: {args.left_rel_dir}")
  logging.info(f"balanced_right_rel_dir: {args.balanced_right_rel_dir}")
  logging.info(f"unbalanced_right_rel_dir: {args.unbalanced_right_rel_dir}")
  logging.info(f"left_pattern: {args.left_pattern}")
  logging.info(f"gt_disp_name: {args.gt_disp_name}")
  logging.info(f"left_occ_mask_name: {args.left_occ_mask_name}")
  logging.info(f"class_ids: {list(args.class_ids)}")
  logging.info(f"valid_iters: {args.valid_iters}")
  logging.info(f"max_disp: {args.max_disp}")
  logging.info(f"low_memory: {args.low_memory}")
  logging.info(f"scale: {args.scale}")
  logging.info(f"paper_protocol: {args.paper_protocol}")
  logging.info(f"balanced_input_scale: {args.balanced_input_scale}")
  logging.info(f"unbalanced_input_scale: {args.unbalanced_input_scale}")
  logging.info(f"benchmark_gt_resolution: {args.benchmark_gt_resolution}")
  logging.info(f"match_unbalanced_left_to_right: {args.match_unbalanced_left_to_right}")
  logging.info(f"out_dir: {args.out_dir}")
  logging.info("metric protocol: dataset-level pixel aggregation over GT-valid pixels")
  logging.info("note: NoOcc tracks are only computed when mask_00.png is available")

  total_samples = 0
  for setup in args.setups:
    setup_dir = os.path.join(split_root, setup)
    if not os.path.isdir(setup_dir):
      logging.warning(f"Skip missing setup dir: {setup_dir}")
      continue

    scene_names = sorted([name for name in os.listdir(setup_dir) if os.path.isdir(os.path.join(setup_dir, name))])
    if args.scene_names is not None:
      scene_name_set = set(args.scene_names)
      scene_names = [name for name in scene_names if name in scene_name_set]
    logging.info(f"[{setup}] scenes: {len(scene_names)}")
    scene_summary[setup] = {}

    for scene_name in scene_names:
      scene_dir = os.path.join(setup_dir, scene_name)
      samples = collect_scene_samples(scene_dir, setup, args)
      if args.max_samples_per_scene is not None:
        samples = samples[:args.max_samples_per_scene]
      scene_stats = init_aggregates(track_names)
      logging.info(f"[{setup}/{scene_name}] samples: {len(samples)}")

      gt_disp = np.load(samples[0]["gt_file"]).astype(np.float32)
      occ_mask = None
      if samples[0]["occ_mask_file"] is not None:
        occ_mask = read_mask_png(samples[0]["occ_mask_file"]) > 0
      material_mask = None
      if samples[0]["material_mask_file"] is not None:
        material_mask = read_mask_png(samples[0]["material_mask_file"])
      validate_scene_annotations(gt_disp, occ_mask, material_mask, setup, scene_name)

      for sample_id, sample in enumerate(samples, start=1):
        logging.info(f"[{setup}/{scene_name}] {sample_id}/{len(samples)} -> {sample['left_name']}")
        pred_disp, pred_meta = infer_single_pair(
            model,
            args,
            setup,
            sample["left_file"],
            sample["right_file"],
            return_native=True,
        )
        if pred_disp.shape != gt_disp.shape:
          raise ValueError(
              f"Prediction shape {pred_disp.shape} does not match GT shape {gt_disp.shape} for {setup}/{scene_name}/{sample['left_name']}"
          )

        valid_mask = np.isfinite(gt_disp) & (gt_disp > 0.0)
        track_masks = compute_track_masks(valid_mask, occ_mask, material_mask, track_defs)

        row = {
            "setup": setup,
            "scene": scene_name,
            "sample": sample["left_name"],
            "num_valid_pixels": int(valid_mask.sum()),
        }

        abs_err = np.abs(pred_disp - gt_disp)

        if args.benchmark_gt_resolution == "quarter":
          if setup == "balanced" and abs(float(pred_meta["total_scale"]) - 0.25) < 1e-6:
            gt_metric = downsample_map_linear(gt_disp, 0.25, value_scale=0.25)
            pred_metric = pred_meta["native_disp"]
            valid_mask_metric = downsample_mask_nearest(valid_mask, 0.25) > 0
            occ_mask_metric = None if occ_mask is None else (downsample_mask_nearest(occ_mask, 0.25) > 0)
            material_mask_metric = None if material_mask is None else downsample_mask_nearest(material_mask, 0.25)
          else:
            gt_metric = downsample_map_linear(gt_disp, 0.25, value_scale=0.25)
            pred_metric = downsample_map_linear(pred_disp, 0.25, value_scale=0.25)
            valid_mask_metric = downsample_mask_nearest(valid_mask, 0.25) > 0
            occ_mask_metric = None if occ_mask is None else (downsample_mask_nearest(occ_mask, 0.25) > 0)
            material_mask_metric = None if material_mask is None else downsample_mask_nearest(material_mask, 0.25)
          gt_resolution_name = "Quarter"
        else:
          gt_metric = gt_disp
          pred_metric = pred_disp
          valid_mask_metric = valid_mask
          occ_mask_metric = occ_mask
          material_mask_metric = material_mask
          gt_resolution_name = "Full"

        metric_track_masks = compute_track_masks(valid_mask_metric, occ_mask_metric, material_mask_metric, track_defs)
        abs_err_metric = np.abs(pred_metric - gt_metric)
        for track in track_defs:
          metric_mask = metric_track_masks[track["name"]]
          if metric_mask is None:
            continue
          metric_errors = abs_err_metric[metric_mask]
          if track["kind"] == "all":
            occ_name = "All"
            class_name = "All"
          elif track["kind"] == "noc":
            occ_name = "No Occ"
            class_name = "All"
          elif track["kind"] == "class":
            occ_name = "All"
            class_name = str(track["class_id"])
          else:
            occ_name = "No Occ"
            class_name = str(track["class_id"])
          append_benchmark_row(
              benchmark_quarter_rows,
              setup=setup,
              split_name=args.split_dir_name,
              occlusion=occ_name,
              gt_resolution=gt_resolution_name,
              cls_name=class_name,
              errors=metric_errors,
          )

        for track_name in track_names:
          track_mask = track_masks[track_name]
          if track_mask is None:
            row[f"{track_name}_available"] = 0
            continue
          errors = abs_err[track_mask]
          row[f"{track_name}_available"] = int(errors.size > 0)
          row[f"{track_name}_num_pixels"] = int(errors.size)
          if errors.size == 0:
            continue
          row[f"{track_name}_bad_2"] = float((errors > 2.0).mean() * 100.0)
          row[f"{track_name}_bad_4"] = float((errors > 4.0).mean() * 100.0)
          row[f"{track_name}_bad_6"] = float((errors > 6.0).mean() * 100.0)
          row[f"{track_name}_bad_8"] = float((errors > 8.0).mean() * 100.0)
          row[f"{track_name}_mae"] = float(errors.mean())
          row[f"{track_name}_rmse"] = float(np.sqrt(np.square(errors).mean()))
          update_stats(global_stats, track_name, errors)
          update_stats(setup_stats[setup], track_name, errors)
          update_stats(scene_stats, track_name, errors)

        pred_path = save_pred_if_needed(pred_disp, sample, setup, scene_name, args)
        if pred_path is not None:
          row["pred_path"] = pred_path

        sample_rows.append(row)
        total_samples += 1

      scene_summary[setup][scene_name] = finalize_stats(scene_stats)
      log_track_summary(f"{setup}/{scene_name}", scene_summary[setup][scene_name], ("all", "noc"))

  summary = {
      "protocol": {
          "split_root": split_root,
          "split_dir_name": args.split_dir_name,
          "setups": list(args.setups),
          "class_ids": list(args.class_ids),
          "metric_definition": {
              "bad_2": "|pred - gt| > 2 px",
              "bad_4": "|pred - gt| > 4 px",
              "bad_6": "|pred - gt| > 6 px",
              "bad_8": "|pred - gt| > 8 px",
              "mae": "mean absolute error in px",
              "rmse": "root mean squared error in px",
          },
          "valid_mask": "np.isfinite(gt_disp) & (gt_disp > 0.0)",
          "no_occ_mask": "mask_00.png > 0 when available; otherwise NoOcc tracks are unavailable",
          "class_mask": "mask_cat.png for balanced, warped_mask_cat.png for unbalanced when available",
          "aggregation": "dataset-level over pixels, not image-level mean",
          "warning": "Booster labeled test split has withheld GT, so this script is intended for local train-split evaluation and ablation only.",
      },
      "total_samples": total_samples,
      "overall": finalize_stats(global_stats),
      "by_setup": {setup: finalize_stats(stats) for setup, stats in setup_stats.items()},
      "by_scene": scene_summary,
  }

  summary_path = os.path.join(args.out_dir, "summary.json")
  with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

  args_path = os.path.join(args.out_dir, "args.yaml")
  with open(args_path, "w") as f:
    yaml.safe_dump(OmegaConf.to_container(args, resolve=True), f, sort_keys=True)

  csv_path = os.path.join(args.out_dir, "per_sample_metrics.csv")
  fieldnames = sorted({key for row in sample_rows for key in row.keys()}) if sample_rows else ["setup", "scene", "sample"]
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in sample_rows:
      writer.writerow(row)

  benchmark_quarter_path = os.path.join(args.out_dir, "benchmark_protocol_details.csv")
  benchmark_fieldnames = ["Setup", "Split", "Occlusion", "GT Resolution", "Class", "Num Pixels", "Bad 2", "Bad 4", "Bad 6", "Bad 8", "MAE", "RMSE"]
  with open(benchmark_quarter_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=benchmark_fieldnames)
    writer.writeheader()
    for row in benchmark_quarter_rows:
      writer.writerow(row)

  benchmark_summary_path = os.path.join(args.out_dir, "benchmark_protocol_summary.csv")
  benchmark_summary_rows = aggregate_benchmark_rows(benchmark_quarter_rows)
  with open(benchmark_summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=benchmark_fieldnames)
    writer.writeheader()
    for row in benchmark_summary_rows:
      writer.writerow(row)

  if args.benchmark_gt_resolution == "quarter":
    quarter_details_path = os.path.join(args.out_dir, "benchmark_quarter.csv")
    with open(quarter_details_path, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=benchmark_fieldnames)
      writer.writeheader()
      for row in benchmark_quarter_rows:
        writer.writerow(row)

    quarter_summary_path = os.path.join(args.out_dir, "benchmark_quarter_summary.csv")
    with open(quarter_summary_path, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=benchmark_fieldnames)
      writer.writeheader()
      for row in benchmark_summary_rows:
        writer.writerow(row)

  logging.info(f"Total evaluated samples: {total_samples}")
  logging.info(f"Summary saved to: {summary_path}")
  logging.info(f"Per-sample metrics saved to: {csv_path}")
  logging.info(f"Benchmark-protocol details saved to: {benchmark_quarter_path}")
  logging.info(f"Benchmark-protocol summary saved to: {benchmark_summary_path}")

  for setup in args.setups:
    log_track_summary(f"setup={setup}", summary["by_setup"][setup], ("all", "noc"))
  log_track_summary("overall", summary["overall"], ("all", "noc"))


if __name__ == "__main__":
  main()
