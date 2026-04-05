import argparse
import logging
import os
import shutil
import sys
import zipfile
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
  parser.add_argument("--test_dir_name", default="test", type=str)
  parser.add_argument("--setups", nargs="+", default=["balanced", "unbalanced"])
  parser.add_argument("--left_rel_dir", default="camera_00", type=str)
  parser.add_argument("--left_pattern", default="im*.png", type=str)
  parser.add_argument("--balanced_right_rel_dir", default="camera_02", type=str)
  parser.add_argument("--unbalanced_right_rel_dir", default="camera_01", type=str)
  parser.add_argument("--right_from_left_src", default=None, type=str)
  parser.add_argument("--right_from_left_dst", default=None, type=str)
  parser.add_argument("--out_dir", default=f"{code_dir}/../output_booster_submit", type=str)
  parser.add_argument("--zip_name", default="booster_submission.zip", type=str)
  parser.add_argument("--paper_protocol", default="booster_q", choices=["legacy", "booster_full", "booster_q"])
  parser.add_argument("--overwrite", default=0, type=int)
  parser.add_argument("--skip_existing", default=1, type=int)
  parser.add_argument("--make_zip", default=1, type=int)
  parser.add_argument("--scale", default=None, type=float)
  parser.add_argument("--balanced_input_scale", default=None, type=float)
  parser.add_argument("--unbalanced_input_scale", default=None, type=float)
  parser.add_argument("--match_unbalanced_left_to_right", default=1, type=int)
  parser.add_argument("--hiera", default=0, type=int)
  parser.add_argument("--valid_iters", default=None, type=int)
  parser.add_argument("--max_disp", default=None, type=int)
  parser.add_argument("--low_memory", default=None, type=int)
  parser.add_argument("--scene_names", nargs="+", default=None)
  parser.add_argument("--max_samples_per_scene", default=None, type=int)
  return parser.parse_args()


def resolve_test_root(booster_root, test_dir_name):
  candidates = [
      os.path.join(booster_root, test_dir_name),
      os.path.join(booster_root, test_dir_name.capitalize()),
      booster_root,
  ]
  for candidate in candidates:
    if os.path.isdir(candidate):
      return candidate
  raise FileNotFoundError(f"Cannot find BOOSTER test root under {booster_root}")


def get_right_rel_dir(args, setup):
  if setup == "balanced":
    return args.balanced_right_rel_dir
  if setup == "unbalanced":
    return args.unbalanced_right_rel_dir
  raise ValueError(f"Unknown setup: {setup}")


def collect_scene_samples(scene_dir, setup, args):
  left_dir = os.path.join(scene_dir, args.left_rel_dir)
  right_dir = os.path.join(scene_dir, get_right_rel_dir(args, setup))
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
    samples.append((left_file, right_file, left_name))
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
  if cfg.get("low_memory") is None:
    cfg["low_memory"] = 1 if protocol == "booster_q" else 0
  if cfg.get("max_disp") is None and protocol == "booster_q":
    cfg["max_disp"] = 192
  return cfg


def resize_pair(img0, img1, scale_factor):
  if scale_factor == 1.0:
    return img0, img1
  new_w = max(1, int(round(img0.shape[1] * scale_factor)))
  new_h = max(1, int(round(img0.shape[0] * scale_factor)))
  img0 = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
  img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
  return img0, img1


def infer_single_pair(model, args, setup, left_file, right_file):
  img0 = read_rgb(left_file)
  img1 = read_rgb(right_file)
  ori_h, ori_w = img0.shape[:2]
  disp_scale_x = 1.0

  # Booster paper evaluates balanced at F/H/Q and handles unbalanced by
  # matching the left view to the right-view resolution first.
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
  disp = disp.data.cpu().numpy().reshape(infer_h, infer_w).clip(0, None)

  if disp_scale_x != 1.0 or infer_h != ori_h or infer_w != ori_w:
    disp = cv2.resize(disp, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
    disp = disp / disp_scale_x

  return disp


def save_submission_disp(disp, out_file):
  disp_to_save = np.round(disp * 64.0).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)
  os.makedirs(os.path.dirname(out_file), exist_ok=True)
  cv2.imwrite(out_file, disp_to_save)


def zip_submission(folder, zip_path):
  with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for setup_name in ("balanced", "unbalanced"):
      setup_dir = os.path.join(folder, setup_name)
      if not os.path.isdir(setup_dir):
        continue
      for root, _, files in os.walk(setup_dir):
        for name in sorted(files):
          file_path = os.path.join(root, name)
          arcname = os.path.relpath(file_path, folder)
          zf.write(file_path, arcname)


def main():
  cli_args = parse_args()
  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)

  with open(f"{os.path.dirname(cli_args.model_dir)}/cfg.yaml", "r") as ff:
    cfg = yaml.safe_load(ff)

  for k, v in vars(cli_args).items():
    if k not in cfg or v is not None:
      cfg[k] = v
  cfg = apply_protocol_defaults(cfg)
  args = OmegaConf.create(cfg)

  test_root = resolve_test_root(args.booster_root, args.test_dir_name)
  if os.path.exists(args.out_dir):
    if args.overwrite:
      shutil.rmtree(args.out_dir)
  os.makedirs(args.out_dir, exist_ok=True)

  logging.info(f"booster_root: {args.booster_root}")
  logging.info(f"test_root: {test_root}")
  logging.info(f"model_dir: {args.model_dir}")
  logging.info(f"setups: {list(args.setups)}")
  logging.info(f"left_rel_dir: {args.left_rel_dir}")
  logging.info(f"balanced_right_rel_dir: {args.balanced_right_rel_dir}")
  logging.info(f"unbalanced_right_rel_dir: {args.unbalanced_right_rel_dir}")
  logging.info(f"left_pattern: {args.left_pattern}")
  logging.info(f"valid_iters: {args.valid_iters}")
  logging.info(f"max_disp: {args.max_disp}")
  logging.info(f"low_memory: {args.low_memory}")
  logging.info(f"scale: {args.scale}")
  logging.info(f"paper_protocol: {args.paper_protocol}")
  logging.info(f"balanced_input_scale: {args.balanced_input_scale}")
  logging.info(f"unbalanced_input_scale: {args.unbalanced_input_scale}")
  logging.info(f"match_unbalanced_left_to_right: {args.match_unbalanced_left_to_right}")
  logging.info(f"out_dir: {args.out_dir}")

  model = torch.load(args.model_dir, map_location="cpu", weights_only=False)
  model.args.valid_iters = args.valid_iters
  model.args.max_disp = args.max_disp
  model.args.low_memory = bool(args.low_memory)
  model.cuda().eval()

  total_samples = 0
  for setup in args.setups:
    setup_dir = os.path.join(test_root, setup)
    if not os.path.isdir(setup_dir):
      logging.warning(f"Skip missing setup dir: {setup_dir}")
      continue

    scene_names = sorted([name for name in os.listdir(setup_dir) if os.path.isdir(os.path.join(setup_dir, name))])
    if args.scene_names is not None:
      scene_name_set = set(args.scene_names)
      scene_names = [name for name in scene_names if name in scene_name_set]
    logging.info(f"[{setup}] scenes: {len(scene_names)}")

    for scene_name in scene_names:
      scene_dir = os.path.join(setup_dir, scene_name)
      out_scene_dir = os.path.join(args.out_dir, setup, scene_name)
      os.makedirs(out_scene_dir, exist_ok=True)

      samples = collect_scene_samples(
          scene_dir,
          setup,
          args,
      )
      if args.max_samples_per_scene is not None:
        samples = samples[:args.max_samples_per_scene]
      logging.info(f"[{setup}/{scene_name}] samples: {len(samples)}")

      for sample_id, (left_file, right_file, out_name) in enumerate(samples, start=1):
        out_file = os.path.join(out_scene_dir, out_name)
        if args.skip_existing and os.path.isfile(out_file):
          logging.info(f"Skip existing: {out_file}")
          total_samples += 1
          continue

        logging.info(f"[{setup}/{scene_name}] {sample_id}/{len(samples)} -> {out_name}")
        disp = infer_single_pair(model, args, setup, left_file, right_file)
        save_submission_disp(disp, out_file)
        total_samples += 1

  args_dump_file = os.path.join(os.path.dirname(args.out_dir), f"{os.path.basename(args.out_dir)}_args.yaml")
  with open(args_dump_file, "w") as f:
    yaml.safe_dump(OmegaConf.to_container(args, resolve=True), f, sort_keys=True)

  logging.info(f"Total saved predictions: {total_samples}")
  if args.make_zip:
    zip_path = os.path.join(os.path.dirname(args.out_dir), args.zip_name)
    if os.path.exists(zip_path):
      os.remove(zip_path)
    zip_submission(args.out_dir, zip_path)
    logging.info(f"Submission zip: {zip_path}")


if __name__ == "__main__":
  main()
