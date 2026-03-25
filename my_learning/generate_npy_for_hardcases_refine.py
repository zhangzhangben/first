import os
import sys
import argparse
import cv2
import torch
import numpy as np
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

# 配置环境路径，确保能找到 core
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from core.utils.utils import InputPadder


AMP_DTYPE = torch.float16


def load_model_and_cfg(model_path, valid_iters, max_disp_override=None):
    loaded = torch.load(model_path, map_location='cpu', weights_only=False)

    if isinstance(loaded, dict):
        raise RuntimeError(
            f"{model_path} 看起来不是完整序列化模型，请优先使用 model_best_bp2_serialize.pth"
        )

    model = loaded
    model.cuda().eval()

    # 优先保留模型自带 args，避免 cfg.yaml 不完整把参数覆盖坏
    if hasattr(model, 'args') and model.args is not None:
        model.args.valid_iters = valid_iters
        if max_disp_override is not None:
            model.args.max_disp = max_disp_override
    else:
        cfg_path = os.path.join(os.path.dirname(model_path), 'cfg.yaml')
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

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

    # 和官方 run_demo 对齐
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
    pred_disp = pred_disp.clamp(min=0)
    return pred_disp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='weights/23-36-37/model_best_bp2_serialize.pth',
        help='模型权重路径'
    )
    parser.add_argument(
        '--hard_root',
        type=str,
        default='./Dataset_Split_Final/Hard_Test_100',
        help='Hard Cases 根目录'
    )
    parser.add_argument(
        '--valid_iters',
        type=int,
        default=8,
        help='推理迭代次数'
    )
    parser.add_argument(
        '--max_disp',
        type=int,
        default=None,
        help='默认使用模型自带配置，不建议乱改'
    )
    parser.add_argument(
        '--overwrite',
        type=int,
        default=0,
        help='是否覆盖已有 npy，0: 跳过已有文件, 1: 强制重算'
    )
    parser.add_argument(
        '--only_name',
        type=str,
        default=None,
        help='只跑某一张图的名字或前缀，例如 0005 或 0005.jpg'
    )
    args = parser.parse_args()

    out_npy_dir = os.path.join(args.hard_root, 'preds_npy')
    os.makedirs(out_npy_dir, exist_ok=True)

    left_dir = os.path.join(args.hard_root, 'left/rgb')
    right_dir = os.path.join(args.hard_root, 'right/rgb')

    img_names = sorted([
        f for f in os.listdir(left_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if args.only_name is not None:
        target_prefix = os.path.splitext(args.only_name)[0]
        img_names = [f for f in img_names if os.path.splitext(f)[0] == target_prefix]

    if not img_names:
        print(f"❌ 找不到图片，请检查路径: {left_dir}")
        return

    print(">>> [1/2] 正在加载模型...")
    model, model_args = load_model_and_cfg(
        args.model_dir,
        valid_iters=args.valid_iters,
        max_disp_override=args.max_disp
    )
    print(f"    使用配置: valid_iters={model_args.valid_iters}, max_disp={model_args.max_disp}")

    print(f">>> [2/2] 开始为 {len(img_names)} 张 Hard Cases 生成 .npy 预测文件...")
    success_count = 0
    skip_count = 0

    with torch.no_grad():
        for name in tqdm(img_names):
            prefix = os.path.splitext(name)[0]
            out_path = os.path.join(out_npy_dir, f"{prefix}.npy")

            if os.path.exists(out_path) and not args.overwrite:
                skip_count += 1
                continue

            left_path = os.path.join(left_dir, name)
            right_candidates = [
                os.path.join(right_dir, name),
                os.path.join(right_dir, prefix + '.png'),
                os.path.join(right_dir, prefix + '.jpg'),
                os.path.join(right_dir, prefix + '.jpeg'),
            ]
            right_path = None
            for cand in right_candidates:
                if os.path.exists(cand):
                    right_path = cand
                    break

            if right_path is None:
                print(f"\n❌ 跳过 {name}: 右图不存在")
                continue

            imgL_bgr = cv2.imread(left_path)
            imgR_bgr = cv2.imread(right_path)

            if imgL_bgr is None or imgR_bgr is None:
                print(f"\n❌ 跳过 {name}: 左右图读取失败")
                continue

            imgL = cv2.cvtColor(imgL_bgr, cv2.COLOR_BGR2RGB)
            imgR = cv2.cvtColor(imgR_bgr, cv2.COLOR_BGR2RGB)

            try:
                pred_disp = run_inference(model, imgL, imgR, model_args.valid_iters)
                np.save(out_path, pred_disp.cpu().numpy())
                success_count += 1
            except Exception as e:
                print(f"\n❌ 跳过 {name}: {e}")

    print(f"\n✅ 完成")
    print(f"保存目录: {out_npy_dir}")
    print(f"成功生成: {success_count}")
    print(f"跳过已有: {skip_count}")


if __name__ == '__main__':
    main()
