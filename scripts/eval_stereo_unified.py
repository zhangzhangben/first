import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

import Utils as U
from core.stereo_datasets import build_dataloader, build_dataset, load_scene_split
from core.utils.utils import InputPadder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--model_dir", default=None, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--subset", default="val", choices=["train", "val"])
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--scene_names", nargs="+", default=None)
    parser.add_argument("--setups", nargs="+", default=None)
    return parser.parse_args()


def load_protocol(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluation_boundary(dataset_name):
    if dataset_name == "booster":
        return {
            "evaluation_scope": "local_formal_eval",
            "official_submission_equivalent": False,
            "claim_boundary": "Use for local heldout comparison and analysis only unless separately submitted to official test server.",
        }
    return {
        "evaluation_scope": "local_heldout_eval",
        "official_submission_equivalent": False,
        "claim_boundary": "This is local heldout evaluation only. It is not an official benchmark submission and must not be described as one.",
    }


def save_boundary_note(path, dataset_name, subset, protocol, args):
    boundary = evaluation_boundary(dataset_name)
    lines = [
        f"# Evaluation Boundary",
        "",
        f"- dataset: `{dataset_name}`",
        f"- subset: `{subset}`",
        f"- evaluation_scope: `{boundary['evaluation_scope']}`",
        f"- official_submission_equivalent: `{boundary['official_submission_equivalent']}`",
        f"- claim_boundary: {boundary['claim_boundary']}",
    ]
    if dataset_name == "booster":
        lines.extend(
            [
                f"- delegated_script: `{protocol['validation']['independent_eval_script']}`",
                f"- split_file: `{protocol['data']['split_file']}`",
                f"- benchmark_gt_resolution: `{protocol['validation']['benchmark_gt_resolution']}`",
                f"- max_disp: `{protocol['validation']['max_disp']}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- independent_eval_script: `{protocol['validation']['independent_eval_script']}`",
                f"- reference_metric_threshold: dataset-specific local threshold inside `eval_stereo_unified.py`",
                f"- note: official dataset-specific submission/eval flow is still pending integration",
            ]
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def configure_model_for_eval(model, protocol):
    val_cfg = protocol["validation"]
    train_cfg = protocol["train"]
    if not hasattr(model, "args") or model.args is None:
        raise RuntimeError("Serialized model is missing model.args; cannot configure eval.")
    model.args.valid_iters = int(val_cfg["valid_iters"])
    model.args.max_disp = int(val_cfg["max_disp"])
    model.args.low_memory = int(val_cfg["low_memory"])
    model.args.mixed_precision = bool(train_cfg["amp_dtype"] != "fp32")
    return model


def infer_batch(model, batch, protocol):
    left = batch["left"].cuda(non_blocking=True).unsqueeze(0)
    right = batch["right"].cuda(non_blocking=True).unsqueeze(0)
    gt = batch["disp"].cuda(non_blocking=True).unsqueeze(0)
    valid = batch["valid"].cuda(non_blocking=True).unsqueeze(0)
    padder = InputPadder(left.shape, divis_by=32, force_square=False)
    left, right = padder.pad(left, right)
    with torch.amp.autocast(
        "cuda",
        enabled=protocol["train"]["amp_dtype"] != "fp32",
        dtype=U.AMP_DTYPE,
    ):
        pred = model.forward(
            left,
            right,
            iters=int(protocol["validation"]["valid_iters"]),
            test_mode=True,
            low_memory=bool(protocol["validation"]["low_memory"]),
            optimize_build_volume="pytorch1",
        )
    pred = padder.unpad(pred.float())
    return pred, gt, valid


def reference_threshold_for_dataset(dataset_name):
    if dataset_name in {"sceneflow", "eth3d"}:
        return 1.0
    if dataset_name == "middlebury":
        return 2.0
    if dataset_name.startswith("kitti"):
        return 3.0
    return 2.0


def summarize_error_array(errors, threshold):
    if errors.size == 0:
        return {
            "available": False,
            "num_pixels": 0,
        }
    return {
        "available": True,
        "num_pixels": int(errors.size),
        "epe": float(errors.mean()),
        "mae": float(errors.mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "bad_1": float((errors > 1.0).mean() * 100.0),
        "bad_2": float((errors > 2.0).mean() * 100.0),
        "bad_3": float((errors > 3.0).mean() * 100.0),
        "bad_5": float((errors > 5.0).mean() * 100.0),
        "reference_threshold": float(threshold),
        "reference_bad": float((errors > float(threshold)).mean() * 100.0),
    }


def update_aggregate(store, key, errors, noc_errors=None):
    item = store.setdefault(
        key,
        {
            "num_pixels": 0,
            "sum_abs": 0.0,
            "sum_sq": 0.0,
            "bad_1_count": 0,
            "bad_2_count": 0,
            "bad_3_count": 0,
            "bad_5_count": 0,
            "num_samples": 0,
            "noc_num_pixels": 0,
            "noc_sum_abs": 0.0,
            "noc_sum_sq": 0.0,
            "noc_bad_1_count": 0,
            "noc_bad_2_count": 0,
            "noc_bad_3_count": 0,
            "noc_bad_5_count": 0,
        },
    )
    if errors.size > 0:
        item["num_pixels"] += int(errors.size)
        item["sum_abs"] += float(errors.sum())
        item["sum_sq"] += float(np.square(errors).sum())
        item["bad_1_count"] += int((errors > 1.0).sum())
        item["bad_2_count"] += int((errors > 2.0).sum())
        item["bad_3_count"] += int((errors > 3.0).sum())
        item["bad_5_count"] += int((errors > 5.0).sum())
        item["num_samples"] += 1
    if noc_errors is not None and noc_errors.size > 0:
        item["noc_num_pixels"] += int(noc_errors.size)
        item["noc_sum_abs"] += float(noc_errors.sum())
        item["noc_sum_sq"] += float(np.square(noc_errors).sum())
        item["noc_bad_1_count"] += int((noc_errors > 1.0).sum())
        item["noc_bad_2_count"] += int((noc_errors > 2.0).sum())
        item["noc_bad_3_count"] += int((noc_errors > 3.0).sum())
        item["noc_bad_5_count"] += int((noc_errors > 5.0).sum())


def finalize_aggregate(store, threshold):
    out = {}
    for key, item in store.items():
        num_pixels = item["num_pixels"]
        if num_pixels > 0:
            out[key] = {
                "available": True,
                "num_samples": int(item["num_samples"]),
                "num_pixels": int(num_pixels),
                "epe": item["sum_abs"] / num_pixels,
                "mae": item["sum_abs"] / num_pixels,
                "rmse": float(np.sqrt(item["sum_sq"] / num_pixels)),
                "bad_1": item["bad_1_count"] * 100.0 / num_pixels,
                "bad_2": item["bad_2_count"] * 100.0 / num_pixels,
                "bad_3": item["bad_3_count"] * 100.0 / num_pixels,
                "bad_5": item["bad_5_count"] * 100.0 / num_pixels,
                "reference_threshold": float(threshold),
                "reference_bad": (item["bad_1_count"] if threshold == 1 else item["bad_2_count"] if threshold == 2 else item["bad_3_count"] if threshold == 3 else item["bad_5_count"]) * 100.0 / num_pixels if threshold in {1.0, 2.0, 3.0, 5.0} else None,
            }
        else:
            out[key] = {"available": False, "num_samples": 0, "num_pixels": 0}
        noc_num_pixels = item["noc_num_pixels"]
        if noc_num_pixels > 0:
            out[key]["noc"] = {
                "available": True,
                "num_pixels": int(noc_num_pixels),
                "epe": item["noc_sum_abs"] / noc_num_pixels,
                "mae": item["noc_sum_abs"] / noc_num_pixels,
                "rmse": float(np.sqrt(item["noc_sum_sq"] / noc_num_pixels)),
                "bad_1": item["noc_bad_1_count"] * 100.0 / noc_num_pixels,
                "bad_2": item["noc_bad_2_count"] * 100.0 / noc_num_pixels,
                "bad_3": item["noc_bad_3_count"] * 100.0 / noc_num_pixels,
                "bad_5": item["noc_bad_5_count"] * 100.0 / noc_num_pixels,
            }
        else:
            out[key]["noc"] = {"available": False, "num_pixels": 0}
    return out


def save_rows_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dispatch_booster(protocol, args):
    data_cfg = protocol["data"]
    val_cfg = protocol["validation"]
    model_dir = args.model_dir or protocol["model"]["model_dir"]
    os.makedirs(args.out_dir, exist_ok=True)

    scene_names = args.scene_names
    if scene_names is None:
        split_cfg = load_scene_split(data_cfg["split_file"])
        scene_names = split_cfg["train_scenes"] if args.subset == "train" else split_cfg["val_scenes"]

    cmd = [
        sys.executable,
        os.path.join(code_dir, "eval_booster.py"),
        "--model_dir",
        model_dir,
        "--booster_root",
        data_cfg["booster_root"],
        "--split_dir_name",
        data_cfg["split_dir_name"],
        "--out_dir",
        args.out_dir,
        "--paper_protocol",
        str(val_cfg["paper_protocol"]),
        "--scale",
        str(data_cfg["scale"]),
        "--balanced_input_scale",
        str(data_cfg["balanced_input_scale"]),
        "--unbalanced_input_scale",
        str(data_cfg["unbalanced_input_scale"]),
        "--match_unbalanced_left_to_right",
        str(int(data_cfg["match_unbalanced_left_to_right"])),
        "--benchmark_gt_resolution",
        str(val_cfg["benchmark_gt_resolution"]),
        "--hiera",
        str(int(val_cfg["hiera"])),
        "--valid_iters",
        str(int(val_cfg["valid_iters"])),
        "--max_disp",
        str(int(val_cfg["max_disp"])),
        "--low_memory",
        str(int(val_cfg["low_memory"])),
    ]
    setups = args.setups or data_cfg.get("val_setups", ["balanced", "unbalanced"])
    if setups:
        cmd += ["--setups", *list(setups)]
    if scene_names:
        cmd += ["--scene_names", *list(scene_names)]
    if args.max_samples is not None:
        cmd += ["--max_samples_per_scene", str(int(args.max_samples))]

    with open(os.path.join(args.out_dir, "delegated_command.sh"), "w") as f:
        f.write(" ".join(shlex.quote(x) for x in cmd) + "\n")
    save_boundary_note(os.path.join(args.out_dir, "evaluation_boundary.md"), "booster", args.subset, protocol, args)

    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    protocol = load_protocol(args.config)
    dataset_name = protocol["data"]["dataset"].lower()
    model_dir = args.model_dir or protocol["model"]["model_dir"]

    U.set_logging_format()
    if dataset_name == "booster":
        dispatch_booster(protocol, args)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "command.sh"), "w") as f:
        f.write(" ".join(shlex.quote(x) for x in sys.argv) + "\n")
    with open(os.path.join(args.out_dir, "protocol_resolved.yaml"), "w") as f:
        yaml.safe_dump(protocol, f, sort_keys=False)
    save_boundary_note(os.path.join(args.out_dir, "evaluation_boundary.md"), dataset_name, args.subset, protocol, args)

    model = torch.load(model_dir, map_location="cpu", weights_only=False)
    model = configure_model_for_eval(model, protocol)
    model.cuda().eval()

    dataset = build_dataset(dict(protocol["data"]), subset=args.subset)
    loader = build_dataloader(dataset, batch_size=1, num_workers=0, shuffle=False)
    threshold = reference_threshold_for_dataset(dataset_name)

    overall = {}
    by_scene = {}
    by_setup = {}
    per_sample_rows = []

    processed = 0
    for batch in loader:
        if args.scene_names is not None and batch["scene"] not in set(args.scene_names):
            continue
        pred, gt, valid = infer_batch(model, batch, protocol)
        pred_np = pred.squeeze().detach().cpu().numpy().astype(np.float32)
        gt_np = gt.squeeze().detach().cpu().numpy().astype(np.float32)
        valid_np = valid.squeeze().detach().cpu().numpy() > 0.5
        errors = np.abs(pred_np - gt_np)[valid_np]

        noc_errors = None
        if "noc" in batch:
            noc_np = batch["noc"].squeeze().detach().cpu().numpy() > 0.5
            noc_mask = valid_np & noc_np
            noc_errors = np.abs(pred_np - gt_np)[noc_mask]

        update_aggregate(overall, "all", errors, noc_errors)
        update_aggregate(by_scene, batch["scene"], errors, noc_errors)
        update_aggregate(by_setup, batch["setup"], errors, noc_errors)

        sample_metrics = summarize_error_array(errors, threshold)
        row = {
            "dataset": batch["dataset"],
            "setup": batch["setup"],
            "scene": batch["scene"],
            "name": batch["name"],
            "left_file": batch["left_file"],
            "right_file": batch["right_file"],
            "num_pixels": sample_metrics["num_pixels"],
            "epe": sample_metrics.get("epe"),
            "rmse": sample_metrics.get("rmse"),
            "bad_1": sample_metrics.get("bad_1"),
            "bad_2": sample_metrics.get("bad_2"),
            "bad_3": sample_metrics.get("bad_3"),
            "bad_5": sample_metrics.get("bad_5"),
            "reference_threshold": sample_metrics.get("reference_threshold"),
            "reference_bad": sample_metrics.get("reference_bad"),
        }
        if noc_errors is not None:
            noc_metrics = summarize_error_array(noc_errors, threshold)
            row.update(
                {
                    "noc_num_pixels": noc_metrics["num_pixels"],
                    "noc_epe": noc_metrics.get("epe"),
                    "noc_bad_1": noc_metrics.get("bad_1"),
                    "noc_bad_2": noc_metrics.get("bad_2"),
                    "noc_bad_3": noc_metrics.get("bad_3"),
                    "noc_bad_5": noc_metrics.get("bad_5"),
                }
            )
        per_sample_rows.append(row)

        processed += 1
        if args.max_samples is not None and processed >= int(args.max_samples):
            break

    summary = {
        "dataset": dataset_name,
        "subset": args.subset,
        "model_dir": model_dir,
        **evaluation_boundary(dataset_name),
        "notes": [
            "Read evaluation_boundary.md before using these results in reports or papers.",
            "Do not mix local heldout evaluation with official benchmark submission claims.",
        ],
        "reference_threshold": threshold,
        "overall": finalize_aggregate(overall, threshold),
        "by_scene": finalize_aggregate(by_scene, threshold),
        "by_setup": finalize_aggregate(by_setup, threshold),
        "num_samples": len(per_sample_rows),
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fieldnames = [
        "dataset",
        "setup",
        "scene",
        "name",
        "left_file",
        "right_file",
        "num_pixels",
        "epe",
        "rmse",
        "bad_1",
        "bad_2",
        "bad_3",
        "bad_5",
        "reference_threshold",
        "reference_bad",
        "noc_num_pixels",
        "noc_epe",
        "noc_bad_1",
        "noc_bad_2",
        "noc_bad_3",
        "noc_bad_5",
    ]
    save_rows_csv(os.path.join(args.out_dir, "per_sample_metrics.csv"), per_sample_rows, fieldnames)


if __name__ == "__main__":
    main()
