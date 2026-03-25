import os
import sys
import shutil
import argparse
import cv2
import torch
import numpy as np
import imageio.v2 as imageio
import yaml
from omegaconf import OmegaConf



# ---------------------------------------------------------
# 1. 环境配置
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from core.utils.utils import InputPadder
from Utils import set_seed, vis_disparity


AMP_DTYPE = torch.float16


# ---------------------------------------------------------
# 2. 核心辅助函数
# ---------------------------------------------------------
def depth_uint8_decoding(depth_uint8, scale=1000):
    depth_uint8 = depth_uint8.astype(np.float32)
    out = depth_uint8[..., 0] * 255 * 255 + depth_uint8[..., 1] * 255 + depth_uint8[..., 2]
    return out / float(scale)


def compute_metrics(pred_disp, gt_disp):
    mask = (gt_disp > 0.1) & (gt_disp < 1024)

    H, W = gt_disp.shape
    xx = torch.arange(W, device=gt_disp.device).view(1, -1).repeat(H, 1)
    invisible_mask = (xx - gt_disp) < 0
    mask = mask & (~invisible_mask)

    if not mask.any():
        return 0.0, mask

    abs_err = torch.abs(pred_disp[mask] - gt_disp[mask])
    bad_2 = (abs_err > 2.0).float().mean().item() * 100.0
    return bad_2, mask


def save_vis_disp(disp, path, max_disp=None):
    disp_np = disp.squeeze().detach().cpu().numpy()
    vis = vis_disparity(
        disp_np,
        min_val=None,
        max_val=None,
        cmap=None,
        color_map=cv2.COLORMAP_TURBO
    )
    cv2.imwrite(path, vis[:, :, ::-1])


def save_error_map(pred, gt, mask, path):
    err = torch.abs(pred - gt)
    err[~mask] = 0
    err_np = err.detach().cpu().numpy()
    err_vis = (err_np / 5.0 * 255.0).clip(0, 255).astype(np.uint8)
    err_color = cv2.applyColorMap(err_vis, cv2.COLORMAP_HOT)
    cv2.imwrite(path, err_color)


def create_dataset_structure(base_path, is_hard=False):
    sub_dirs = ['left/rgb', 'right/rgb', 'left/disparity']
    if is_hard:
        sub_dirs.append('visualizations')
    for sd in sub_dirs:
        os.makedirs(os.path.join(base_path, sd), exist_ok=True)


def load_model_and_cfg(model_path, valid_iters, max_disp_override=None):
    loaded = torch.load(model_path, map_location='cpu', weights_only=False)

    if isinstance(loaded, dict):
        raise RuntimeError(
            f"{model_path} 看起来不是完整序列化模型，请优先使用 model_best_bp2_serialize.pth"
        )

    model = loaded
    model.cuda().eval()

    # 优先保留模型自带 args，避免 cfg.yaml 不完整导致缺键
    if hasattr(model, 'args') and model.args is not None:
        model.args.valid_iters = valid_iters
        if max_disp_override is not None:
            model.args.max_disp = max_disp_override
    else:
        cfg_path = os.path.join(os.path.dirname(model_path), 'cfg.yaml')
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # 补默认值，防止当前代码访问缺失键
        cfg.setdefault('normalize', True)
        cfg.setdefault('corr_levels', 2)
        cfg.setdefault('corr_radius', 4)
        cfg.setdefault('mixed_precision', True)
        cfg.setdefault('low_memory', 0)
        cfg.setdefault('n_gru_layers', 1)
        cfg.setdefault('slow_fast_gru', False)

        cfg['valid_iters'] = valid_iters
        if max_disp_override is not None:
            cfg['max_disp'] = max_disp_override

        model.args = OmegaConf.create(cfg)

    return model, model.args


def run_inference(model, imgL, imgR, valid_iters):
    tL = torch.as_tensor(imgL).cuda().float()[None].permute(0, 3, 1, 2)
    tR = torch.as_tensor(imgR).cuda().float()[None].permute(0, 3, 1, 2)

    # 与官方 run_demo.py 对齐
    padder = InputPadder(tL.shape, divis_by=32, force_square=False)
    tL_p, tR_p = padder.pad(tL, tR)

    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        pred_p = model.forward(
            tL_p,
            tR_p,
            iters=valid_iters,
            test_mode=True,
            optimize_build_volume='pytorch1'
        )

    pred_disp = padder.unpad(pred_p.float()).squeeze()
    return pred_disp


# ---------------------------------------------------------
# 3. 主程序
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--out_root', default='./Dataset_Split_Final', type=str)
    parser.add_argument('--max_disp', type=int, default=None, help='默认使用模型自带配置')
    parser.add_argument('--valid_iters', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=100)
    args = parser.parse_args()

    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    hard_root = os.path.join(args.out_root, f'Hard_Test_{args.top_k}')
    train_root = os.path.join(args.out_root, 'Train_Remaining')
    create_dataset_structure(hard_root, is_hard=True)
    create_dataset_structure(train_root, is_hard=False)

    print(">>> [1/5] 加载模型中...")
    model, model_args = load_model_and_cfg(
        args.model_dir,
        valid_iters=args.valid_iters,
        max_disp_override=args.max_disp
    )
    print(f"    使用配置: valid_iters={model_args.valid_iters}, max_disp={model_args.max_disp}")

    src_left_rgb = os.path.join(args.data_dir, 'left', 'rgb')
    src_right_rgb = os.path.join(args.data_dir, 'right', 'rgb')
    src_left_disp = os.path.join(args.data_dir, 'left', 'disparity')

    img_names = sorted(
        [f for f in os.listdir(src_left_rgb) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    )

    results = []

    print(f">>> [2/5] 全量推理评估开始 ({len(img_names)}张)...")
    for i, img_name in enumerate(img_names):
        try:
            left_path = os.path.join(src_left_rgb, img_name)
            right_path = os.path.join(src_right_rgb, img_name)
            gt_path = os.path.join(src_left_disp, os.path.splitext(img_name)[0] + '.png')

            imgL_bgr = cv2.imread(left_path)
            imgR_bgr = cv2.imread(right_path)
            if imgL_bgr is None or imgR_bgr is None:
                raise RuntimeError("左右图读取失败")

            imgL = cv2.cvtColor(imgL_bgr, cv2.COLOR_BGR2RGB)
            imgR = cv2.cvtColor(imgR_bgr, cv2.COLOR_BGR2RGB)

            gt_raw = imageio.imread(gt_path)
            gt_t = torch.from_numpy(depth_uint8_decoding(gt_raw)).float().cuda()

            pred_disp = run_inference(model, imgL, imgR, model_args.valid_iters)
            bad2, _ = compute_metrics(pred_disp, gt_t)

            results.append({
                'name': img_name,
                'bad2': bad2
            })

            if (i + 1) % 100 == 0:
                print(f"   进度: {i + 1}/{len(img_names)}")

        except Exception as e:
            print(f"❌ 跳过 {img_name}: {e}")

    if len(results) == 0:
        print("❌ 没有成功评测任何样本，请检查数据路径、模型权重和 GT 格式。")
        return

    avg_bad2 = sum(x['bad2'] for x in results) / len(results)
    print(f"\n📊 全部成功样本的平均 Bad2.0: {avg_bad2:.4f}%")
    print(f"📊 成功评测样本数: {len(results)} / {len(img_names)}")

    results.sort(key=lambda x: x['bad2'], reverse=True)
    hard_list = results[:args.top_k]
    train_list = results[args.top_k:]

    print(">>> [3/5] 开始提取并可视化 Hard Cases (PPT素材生成中)...")
    for i, res in enumerate(hard_list):
        name = res['name']
        prefix = os.path.splitext(name)[0]

        shutil.copy(os.path.join(src_left_rgb, name), os.path.join(hard_root, 'left/rgb', name))
        shutil.copy(os.path.join(src_right_rgb, name), os.path.join(hard_root, 'right/rgb', name))
        shutil.copy(os.path.join(src_left_disp, prefix + '.png'), os.path.join(hard_root, 'left/disparity', prefix + '.png'))

        imgL_bgr = cv2.imread(os.path.join(src_left_rgb, name))
        imgR_bgr = cv2.imread(os.path.join(src_right_rgb, name))
        imgL = cv2.cvtColor(imgL_bgr, cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(imgR_bgr, cv2.COLOR_BGR2RGB)

        gt_t = torch.from_numpy(
            depth_uint8_decoding(imageio.imread(os.path.join(src_left_disp, prefix + '.png')))
        ).float().cuda()

        pred = run_inference(model, imgL, imgR, model_args.valid_iters)

        save_vis_disp(
            pred,
            os.path.join(hard_root, 'visualizations', f"rank{i+1:02d}_{prefix}_pred.png"),
            max_disp=model_args.max_disp
        )

        _, mask = compute_metrics(pred, gt_t)
        save_error_map(
            pred,
            gt_t,
            mask,
            os.path.join(hard_root, 'visualizations', f"rank{i+1:02d}_{prefix}_error.png")
        )

    print(">>> [4/5] 正在搬运剩余训练数据...")
    for res in train_list:
        name = res['name']
        prefix = os.path.splitext(name)[0]

        shutil.copy(os.path.join(src_left_rgb, name), os.path.join(train_root, 'left/rgb', name))
        shutil.copy(os.path.join(src_right_rgb, name), os.path.join(train_root, 'right/rgb', name))
        shutil.copy(os.path.join(src_left_disp, prefix + '.png'), os.path.join(train_root, 'left/disparity', prefix + '.png'))

    print("\n>>> [5/5] 任务完成！")
    print(f"📂 PPT素材位于: {hard_root}/visualizations")
    print(f"📊 训练集(Easy): {train_root}")
    print(f"📊 平均 Bad2.0: {avg_bad2:.4f}%")


if __name__ == '__main__':
    main()
