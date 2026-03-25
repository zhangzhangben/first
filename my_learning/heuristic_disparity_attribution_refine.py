import os
import cv2
import numpy as np
import imageio.v2 as imageio
import pandas as pd
from tqdm import tqdm
import argparse


# ---------------------------------------------------------
# 1. 基础工具函数
# ---------------------------------------------------------
def depth_uint8_decoding(depth_uint8, scale=1000):
    """官方 RGB 编码视差转物理视差"""
    depth_uint8 = depth_uint8.astype(np.float32)
    out = depth_uint8[..., 0] * 255 * 255 + depth_uint8[..., 1] * 255 + depth_uint8[..., 2]
    return out / float(scale)


def build_eval_valid_mask(gt_disp, pred_disp):
    """
    和 mine-refine.py 尽量对齐的有效区域：
    1. GT 有效
    2. 预测值有限且非负
    3. 去掉左边缘不可见区
    """
    gt_valid = (gt_disp > 0.1) & (gt_disp < 1024)
    pred_valid = np.isfinite(pred_disp) & (pred_disp >= 0)

    h, w = gt_disp.shape
    xx = np.tile(np.arange(w), (h, 1))
    invisible_mask = (xx - gt_disp) < 0

    valid_mask = gt_valid & pred_valid & (~invisible_mask)
    return valid_mask, invisible_mask


def get_shadow_mask(gt_disp, gt_valid=None):
    """
    基于 1D 扫描线的几何阴影检测
    只在 GT 有效区域内工作，避免无效值污染
    """
    h, w = gt_disp.shape
    occ_mask = np.zeros_like(gt_disp, dtype=bool)

    if gt_valid is None:
        gt_valid = (gt_disp > 0.1) & (gt_disp < 1024)

    xx = np.tile(np.arange(w), (h, 1))
    occ_mask[(xx - gt_disp < 0) & gt_valid] = True

    for y in range(h):
        max_right_frontier = -1e9
        for x in range(w):
            if not gt_valid[y, x]:
                continue

            curr_proj_x = x - gt_disp[y, x]
            if curr_proj_x < max_right_frontier:
                occ_mask[y, x] = True
            else:
                max_right_frontier = curr_proj_x

    return occ_mask


def get_discontinuity_mask(gt_disp, gt_valid=None, jump_thres=1.0):
    """
    视差不连续区检测
    只在 GT 有效区域附近计算
    """
    if gt_valid is None:
        gt_valid = (gt_disp > 0.1) & (gt_disp < 1024)

    diff_h = np.abs(gt_disp[:, 1:] - gt_disp[:, :-1]) > jump_thres
    diff_v = np.abs(gt_disp[1:, :] - gt_disp[:-1, :]) > jump_thres

    valid_h = gt_valid[:, 1:] & gt_valid[:, :-1]
    valid_v = gt_valid[1:, :] & gt_valid[:-1, :]

    disc_mask = np.zeros_like(gt_disp, dtype=bool)
    disc_mask[:, :-1] |= diff_h & valid_h
    disc_mask[:-1, :] |= diff_v & valid_v

    kernel = np.ones((3, 3), np.uint8)
    disc_mask = cv2.dilate(disc_mask.astype(np.uint8), kernel) > 0
    disc_mask &= gt_valid
    return disc_mask


def get_textureless_mask(imgL, gt_valid=None, threshold=4.0):
    """
    低纹理区检测
    """
    gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (7, 7))
    mean_sq = cv2.blur(gray ** 2, (7, 7))
    std_map = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    text_mask = std_map < threshold
    if gt_valid is not None:
        text_mask &= gt_valid
    return text_mask


# ---------------------------------------------------------
# 2. 启发式归因核心逻辑
# ---------------------------------------------------------
def diagnose_errors(imgL, gt_disp, pred_disp, threshold=2.0):
    """
    执行归因统计：遮挡 > 边界 > 弱纹理 > 其他
    归因只在有效评测区域内进行
    """
    valid_mask, invisible_mask = build_eval_valid_mask(gt_disp, pred_disp)

    if not np.any(valid_mask):
        return None

    error_map = np.abs(pred_disp - gt_disp)
    error_pixels = valid_mask & (error_map > threshold)
    total_err_count = int(np.sum(error_pixels))

    gt_valid = (gt_disp > 0.1) & (gt_disp < 1024)

    shadow_m = get_shadow_mask(gt_disp, gt_valid=gt_valid)
    disc_m = get_discontinuity_mask(gt_disp, gt_valid=gt_valid)
    text_m = get_textureless_mask(imgL, gt_valid=gt_valid)

    # 统一限制到评测有效区
    shadow_m &= valid_mask
    disc_m &= valid_mask
    text_m &= valid_mask

    occ_err = error_pixels & shadow_m
    disc_err = error_pixels & (~shadow_m) & disc_m
    text_err = error_pixels & (~shadow_m) & (~disc_m) & text_m
    other_err = error_pixels & (~shadow_m) & (~disc_m) & (~text_m)

    occ_region = valid_mask & shadow_m
    disc_region = valid_mask & (~shadow_m) & disc_m
    text_region = valid_mask & (~shadow_m) & (~disc_m) & text_m
    other_region = valid_mask & (~shadow_m) & (~disc_m) & (~text_m)

    results = {
        "valid_pixels": int(np.sum(valid_mask)),
        "total_errors": total_err_count,
        "bad2_rate": float(total_err_count / (np.sum(valid_mask) + 1e-6)),
        "occlusion": {"count": int(np.sum(occ_err)), "region_size": int(np.sum(occ_region))},
        "discontinuity": {"count": int(np.sum(disc_err)), "region_size": int(np.sum(disc_region))},
        "textureless": {"count": int(np.sum(text_err)), "region_size": int(np.sum(text_region))},
        "others": {"count": int(np.sum(other_err)), "region_size": int(np.sum(other_region))}
    }

    diag_vis = np.zeros((gt_disp.shape[0], gt_disp.shape[1], 3), dtype=np.uint8)
    diag_vis[invisible_mask] = [128, 128, 128]            # 灰色: 不可见区
    diag_vis[shadow_m] = [255, 0, 0]                      # 蓝色: 遮挡
    diag_vis[disc_m & (~shadow_m)] = [0, 255, 0]         # 绿色: 边界
    diag_vis[text_m & (~shadow_m) & (~disc_m)] = [0, 255, 255]  # 黄色: 弱纹理

    return results, diag_vis


# ---------------------------------------------------------
# 3. 批量执行与报告
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hard_root', required=True, help='Hard_Test_100 根目录')
    parser.add_argument('--pred_dir', required=True, help='存放预测结果 npy 的目录')
    parser.add_argument('--out_csv', default='failure_attribution_report.csv')
    parser.add_argument('--bad_thres', type=float, default=2.0, help='Bad-x 阈值，默认 2.0')
    parser.add_argument('--texture_thres', type=float, default=4.0, help='弱纹理阈值')
    args = parser.parse_args()

    rgb_dir = os.path.join(args.hard_root, "left/rgb")
    gt_dir = os.path.join(args.hard_root, "left/disparity")
    vis_dir = os.path.join(args.hard_root, "diagnostic_masks")
    os.makedirs(vis_dir, exist_ok=True)

    summary_list = []
    img_names = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    skipped_missing_pred = 0
    skipped_invalid = 0

    print(f"正在启动启发式归因诊断，分析对象: {len(img_names)} 张")

    total_valid_pixels = 0
    total_bad_pixels = 0
    total_cat_counts = {
        "occlusion": 0,
        "discontinuity": 0,
        "textureless": 0,
        "others": 0,
    }

    for name in tqdm(img_names):
        prefix = os.path.splitext(name)[0]

        img_path = os.path.join(rgb_dir, name)
        gt_path = os.path.join(gt_dir, prefix + ".png")
        pred_path = os.path.join(args.pred_dir, prefix + ".npy")

        if not os.path.exists(pred_path):
            skipped_missing_pred += 1
            continue

        imgL = cv2.imread(img_path)
        if imgL is None:
            skipped_invalid += 1
            continue

        gt_raw = imageio.imread(gt_path)
        gt_disp = depth_uint8_decoding(gt_raw)
        pred_disp = np.load(pred_path)

        if pred_disp.shape != gt_disp.shape:
            print(f"跳过 {name}: pred shape {pred_disp.shape} != gt shape {gt_disp.shape}")
            skipped_invalid += 1
            continue

        diagnosis_data = diagnose_errors(imgL, gt_disp, pred_disp, threshold=args.bad_thres)
        if diagnosis_data is None:
            skipped_invalid += 1
            continue

        stats, mask_vis = diagnosis_data
        cv2.imwrite(os.path.join(vis_dir, f"{prefix}_diag.png"), mask_vis)

        row = {
            "Image": name,
            "Valid_Pixels": stats["valid_pixels"],
            "Total_Bad_Pixels": stats["total_errors"],
            "Bad2_Rate": round(stats["bad2_rate"], 6),
        }

        for cat in ["occlusion", "discontinuity", "textureless", "others"]:
            data = stats[cat]
            err_count = data["count"]
            total_err = stats["total_errors"] + 1e-6
            row[f"{cat}_Count"] = err_count
            row[f"{cat}_Ratio"] = round(err_count / total_err, 4)
            row[f"{cat}_Inner_Rate"] = round(err_count / (data["region_size"] + 1e-6), 4)

            total_cat_counts[cat] += err_count

        ratios = {k: row[f"{k}_Ratio"] for k in ["occlusion", "discontinuity", "textureless", "others"]}
        row["Primary_Reason"] = max(ratios, key=ratios.get)
        summary_list.append(row)

        total_valid_pixels += stats["valid_pixels"]
        total_bad_pixels += stats["total_errors"]

    df = pd.DataFrame(summary_list)
    df.to_csv(args.out_csv, index=False)

    print(f"\n归因报告已生成: {args.out_csv}")
    print(f"成功分析样本数: {len(summary_list)}")
    print(f"缺少 pred.npy 跳过: {skipped_missing_pred}")
    print(f"无效样本跳过: {skipped_invalid}")

    if len(df) > 0:
        print("\n按样本主因统计:")
        print(df["Primary_Reason"].value_counts(normalize=True))

        print("\n按坏点总量加权统计:")
        for cat in ["occlusion", "discontinuity", "textureless", "others"]:
            ratio = total_cat_counts[cat] / (total_bad_pixels + 1e-6)
            print(f"{cat}: {ratio:.4f}")

        print(f"\n总体 Bad2.0: {total_bad_pixels / (total_valid_pixels + 1e-6):.4f}")
    else:
        print("\n没有成功生成任何统计结果，请检查 hard_root 和 pred_dir。")


if __name__ == "__main__":
    main()
