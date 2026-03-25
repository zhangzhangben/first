import os
import sys
import shutil
import argparse
import cv2
import torch
import numpy as np
import imageio.v2 as imageio

# 【1. 环境配置】
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from core.utils.utils import InputPadder
from Utils import set_seed 

AMP_DTYPE = torch.float16 

# ---------------------------------------------------------
# 2. 核心辅助函数
# ---------------------------------------------------------

def depth_uint8_decoding(depth_uint8, scale=1000):
    depth_uint8 = depth_uint8.astype(float)
    out = depth_uint8[...,0]*255*255 + depth_uint8[...,1]*255 + depth_uint8[...,2]
    return out/float(scale)

def compute_metrics(pred_disp, gt_disp):
    mask = (gt_disp > 0.1) & (gt_disp < 1024)
    H, W = gt_disp.shape
    xx = torch.arange(W).view(1, -1).repeat(H, 1).to(gt_disp.device)
    invisible_mask = (xx - gt_disp) < 0
    mask = mask & (~invisible_mask)
    if not mask.any(): return 0.0, mask
    abs_err = torch.abs(pred_disp[mask] - gt_disp[mask])
    bad_2 = (abs_err > 2.0).float().mean().item() * 100.0 
    return bad_2, mask

def save_vis_disp(disp, path, max_disp=192):
    """ 生成彩虹色视差图 """
    disp_np = disp.squeeze().cpu().numpy()
    disp_vis = (disp_np / max_disp * 255.0).clip(0, 255).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
    cv2.imwrite(path, disp_color)

def save_error_map(pred, gt, mask, path):
    """ 生成误差热力图：误差越大越红 """
    err = torch.abs(pred - gt)
    err[~mask] = 0
    err_np = err.cpu().numpy()
    # 归一化误差：0-5像素映射为颜色梯度
    err_vis = (err_np / 5.0 * 255.0).clip(0, 255).astype(np.uint8)
    err_color = cv2.applyColorMap(err_vis, cv2.COLORMAP_HOT)
    cv2.imwrite(path, err_color)

def create_dataset_structure(base_path, is_hard=False):
    sub_dirs = ['left/rgb', 'right/rgb', 'left/disparity']
    if is_hard:
        sub_dirs.append('visualizations') # 专门存放 PPT 素材
    for sd in sub_dirs:
        os.makedirs(os.path.join(base_path, sd), exist_ok=True)

# ---------------------------------------------------------
# 3. 主程序
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--out_root', default='./Dataset_Split_Final', type=str)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--valid_iters', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=100)
    args = parser.parse_args()

    set_seed(0)
    torch.autograd.set_grad_enabled(False) 

    hard_root = os.path.join(args.out_root, f'Hard_Test_{args.top_k}')
    train_root = os.path.join(args.out_root, 'Train_Remaining')
    create_dataset_structure(hard_root, is_hard=True)
    create_dataset_structure(train_root, is_hard=False)

    print(f">>> [1/5] 加载模型中...")
    loaded = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    if isinstance(loaded, dict):
        base_path = os.path.join(os.path.dirname(args.model_dir), 'model_best_bp2_serialize.pth')
        model = torch.load(base_path, map_location='cpu', weights_only=False)
        model.load_state_dict(loaded.get('model', loaded.get('state_dict', loaded)), strict=False)
    else:
        model = loaded
    model.cuda().eval()
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp

    src_left_rgb = os.path.join(args.data_dir, 'left', 'rgb')
    src_right_rgb = os.path.join(args.data_dir, 'right', 'rgb')
    src_left_disp = os.path.join(args.data_dir, 'left', 'disparity')
    img_names = sorted([f for f in os.listdir(src_left_rgb) if f.lower().endswith(('.png', '.jpg'))])
    
    results = []
    
    print(f">>> [2/5] 全量推理评估开始 (2500张)...")
    for i, img_name in enumerate(img_names):
        try:
            imgL = cv2.cvtColor(cv2.imread(os.path.join(src_left_rgb, img_name)), cv2.COLOR_BGR2RGB)
            imgR = cv2.cvtColor(cv2.imread(os.path.join(src_right_rgb, img_name)), cv2.COLOR_BGR2RGB)
            gt_raw = imageio.imread(os.path.join(src_left_disp, os.path.splitext(img_name)[0] + '.png'))
            gt_t = torch.from_numpy(depth_uint8_decoding(gt_raw)).float().cuda()

            tL = torch.as_tensor(imgL).cuda().float()[None].permute(0, 3, 1, 2)
            tR = torch.as_tensor(imgR).cuda().float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(tL.shape, divis_by=32, mode='kitti') 
            tL_p, tR_p = padder.pad(tL, tR)

            with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
                pred_p = model(tL_p, tR_p, iters=args.valid_iters, test_mode=True)
            pred_disp = padder.unpad(pred_p.float()).squeeze()

            bad2, mask = compute_metrics(pred_disp, gt_t)
            results.append({'name': img_name, 'bad2': bad2})
            if (i+1) % 100 == 0: print(f"   进度: {i+1}/2500")
        except Exception as e: print(f"❌ 跳过 {img_name}: {e}")

    results.sort(key=lambda x: x['bad2'], reverse=True)
    hard_list = results[:args.top_k]
    train_list = results[args.top_k:]

    print(f">>> [3/5] 开始提取并可视化 Hard Cases (PPT素材生成中)...")
    for i, res in enumerate(hard_list):
        name = res['name']
        prefix = os.path.splitext(name)[0]
        # 1. 复制原文件
        shutil.copy(os.path.join(src_left_rgb, name), os.path.join(hard_root, 'left/rgb', name))
        shutil.copy(os.path.join(src_right_rgb, name), os.path.join(hard_root, 'right/rgb', name))
        shutil.copy(os.path.join(src_left_disp, prefix + '.png'), os.path.join(hard_root, 'left/disparity', prefix + '.png'))
        
        # 2. 生成可视化 (为了省显存，这里再次推理一次)
        imgL = cv2.cvtColor(cv2.imread(os.path.join(src_left_rgb, name)), cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(cv2.imread(os.path.join(src_right_rgb, name)), cv2.COLOR_BGR2RGB)
        gt_t = torch.from_numpy(depth_uint8_decoding(imageio.imread(os.path.join(src_left_disp, prefix + '.png')))).float().cuda()
        tL = torch.as_tensor(imgL).cuda().float()[None].permute(0, 3, 1, 2)
        tR = torch.as_tensor(imgR).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(tL.shape, divis_by=32, mode='kitti')
        tL_p, tR_p = padder.pad(tL, tR)
        with torch.no_grad():
            pred = padder.unpad(model(tL_p, tR_p, iters=args.valid_iters, test_mode=True)).squeeze()
        
        # 保存彩虹图和误差图
        save_vis_disp(pred, os.path.join(hard_root, 'visualizations', f"rank{i+1:02d}_{prefix}_pred.png"))
        _, mask = compute_metrics(pred, gt_t)
        save_error_map(pred, gt_t, mask, os.path.join(hard_root, 'visualizations', f"rank{i+1:02d}_{prefix}_error.png"))

    print(f">>> [4/5] 正在搬运 2400 张训练数据...")
    for res in train_list:
        name = res['name']
        shutil.copy(os.path.join(src_left_rgb, name), os.path.join(train_root, 'left/rgb', name))
        shutil.copy(os.path.join(src_right_rgb, name), os.path.join(train_root, 'right/rgb', name))
        shutil.copy(os.path.join(src_left_disp, os.path.splitext(name)[0] + '.png'), os.path.join(train_root, 'left/disparity', os.path.splitext(name)[0] + '.png'))

    print(f"\n>>> [5/5] 任务完成！")
    print(f"📂 PPT素材位于: {hard_root}/visualizations")
    print(f"📊 训练集(Easy): {train_root} (2400张)")

if __name__ == '__main__': main()