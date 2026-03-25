import os
import sys
import shutil
import argparse
import cv2
import torch
import numpy as np
import imageio.v2 as imageio

# 【1. 环境配置】确保能够找到 core 和 Utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from core.utils.utils import InputPadder
from Utils import set_seed 

# 显存优化配置
AMP_DTYPE = torch.float16 

# ---------------------------------------------------------
# 2. 核心辅助函数
# ---------------------------------------------------------

def depth_uint8_decoding(depth_uint8, scale=1000):
    """ 官方 RGB -> 物理视差 解码公式 """
    depth_uint8 = depth_uint8.astype(float)
    out = depth_uint8[...,0]*255*255 + depth_uint8[...,1]*255 + depth_uint8[...,2]
    return out/float(scale)

def compute_metrics(pred_disp, gt_disp):
    """ 计算 Bad 2.0，对齐官方物理逻辑（剔除盲区） """
    mask = (gt_disp > 0.1) & (gt_disp < 1024)
    H, W = gt_disp.shape
    xx = torch.arange(W).view(1, -1).repeat(H, 1).to(gt_disp.device)
    invisible_mask = (xx - gt_disp) < 0
    mask = mask & (~invisible_mask)

    if not mask.any(): return 0.0
    abs_err = torch.abs(pred_disp[mask] - gt_disp[mask])
    bad_2 = (abs_err > 2.0).float().mean().item() * 100.0 
    return bad_2

def create_dataset_structure(base_path):
    """ 在目标路径下创建标准数据集结构 """
    sub_dirs = ['left/rgb', 'right/rgb', 'left/disparity']
    for sd in sub_dirs:
        os.makedirs(os.path.join(base_path, sd), exist_ok=True)

# ---------------------------------------------------------
# 3. 主程序：全量评估、排序、双向拆分
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, type=str, help='权重路径 (.pth)')
    parser.add_argument('--data_dir', required=True, type=str, help='原始2500对数据根目录')
    parser.add_argument('--out_root', default='./Dataset_Split', type=str, help='拆分结果的总根目录')
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--valid_iters', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=100, help='筛选出多少张作为Hard Test')
    args = parser.parse_args()

    set_seed(0)
    torch.autograd.set_grad_enabled(False) 

    # --- 1. 初始化两个文件夹的结构 ---
    hard_root = os.path.join(args.out_root, f'Hard_Test_{args.top_k}')
    train_root = os.path.join(args.out_root, 'Train_Remaining')
    
    print(f"\n>>> [1/5] 初始化目录结构...")
    create_dataset_structure(hard_root)
    create_dataset_structure(train_root)

    # --- 2. 加载模型 (兼容混合模式) ---
    print(f">>> [2/5] 正在加载模型进行性能评估...")
    loaded = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    if isinstance(loaded, dict):
        base_path = os.path.join(os.path.dirname(args.model_dir), 'model_best_bp2_serialize.pth')
        if not os.path.exists(base_path): base_path = 'weights/23-36-37/model_best_bp2_serialize.pth'
        model = torch.load(base_path, map_location='cpu', weights_only=False)
        model.load_state_dict(loaded.get('model', loaded.get('state_dict', loaded)), strict=False)
    else:
        model = loaded
    
    model.cuda().eval()
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp

    # --- 3. 准备路径与全量推理 ---
    src_left_rgb = os.path.join(args.data_dir, 'left', 'rgb')
    src_right_rgb = os.path.join(args.data_dir, 'right', 'rgb')
    src_left_disp = os.path.join(args.data_dir, 'left', 'disparity')
    
    img_names = sorted([f for f in os.listdir(src_left_rgb) if f.lower().endswith(('.png', '.jpg'))])
    results = []
    
    print(f">>> [3/5] 开始全量评估 {len(img_names)} 张样本...")
    for i, img_name in enumerate(img_names):
        try:
            # 读取
            imgL_path = os.path.join(src_left_rgb, img_name)
            imgR_path = os.path.join(src_right_rgb, img_name)
            gt_path = os.path.join(src_left_disp, os.path.splitext(img_name)[0] + '.png')
            
            imgL = cv2.cvtColor(cv2.imread(imgL_path), cv2.COLOR_BGR2RGB)
            imgR = cv2.cvtColor(cv2.imread(imgR_path), cv2.COLOR_BGR2RGB)
            gt_raw = imageio.imread(gt_path)
            gt_t = torch.from_numpy(depth_uint8_decoding(gt_raw)).float().cuda()

            # 推理
            tL = torch.as_tensor(imgL).cuda().float()[None].permute(0, 3, 1, 2)
            tR = torch.as_tensor(imgR).cuda().float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(tL.shape, divis_by=32, mode='kitti') 
            tL_p, tR_p = padder.pad(tL, tR)

            with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
                pred_p = model(tL_p, tR_p, iters=args.valid_iters, test_mode=True)
            pred_disp = padder.unpad(pred_p.float()).squeeze()

            # 指标记录
            bad2 = compute_metrics(pred_disp, gt_t)
            results.append({'name': img_name, 'bad2': bad2})
            
            if (i+1) % 100 == 0:
                print(f"   进度: {i+1}/{len(img_names)} | Avg Bad 2.0: {np.mean([x['bad2'] for x in results]):.2f}%")
        except Exception as e:
            print(f"❌ 跳过 {img_name}: {e}")

    # --- 4. 排序并拆分数据集 ---
    print(f"\n>>> [4/5] 正在按误差降序排列并拆分...")
    results.sort(key=lambda x: x['bad2'], reverse=True)
    
    hard_list = results[:args.top_k]
    train_list = results[args.top_k:]
    
    print(f"   🔥 Hard Cases 最差误差: {hard_list[0]['bad2']:.2f}%")
    print(f"   🔥 Hard Cases 最好误差: {hard_list[-1]['bad2']:.2f}%")
    print(f"   ✅ 训练集(剩余部分) 平均误差: {np.mean([x['bad2'] for x in train_list]):.2f}%")

    # --- 5. 执行物理搬运 (复制文件) ---
    def copy_files(file_list, target_root):
        for res in file_list:
            name = res['name']
            disp_name = os.path.splitext(name)[0] + '.png'
            shutil.copy(os.path.join(src_left_rgb, name), os.path.join(target_root, 'left/rgb', name))
            shutil.copy(os.path.join(src_right_rgb, name), os.path.join(target_root, 'right/rgb', name))
            shutil.copy(os.path.join(src_left_disp, disp_name), os.path.join(target_root, 'left/disparity', disp_name))

    print(f">>> [5/5] 正在执行文件搬运...")
    copy_files(hard_list, hard_root)
    print(f"   已完成 Hard Test 集 (100对)")
    copy_files(train_list, train_root)
    print(f"   已完成 Train Remaining 集 ({len(train_list)}对)")

    print(f"\n🎉 拆分圆满完成！")
    print(f"📂 测试集(Hard): {hard_root}")
    print(f"📂 训练集(Easy): {train_root}")

if __name__ == '__main__':
    main()