import torch
import os
import sys

# 当前脚本所在目录
code_dir = os.path.dirname(os.path.realpath(__file__))

# 项目根目录（my_learning 的上一层）
root_dir = os.path.dirname(code_dir)

# 把项目根目录加入 Python 搜索路径
sys.path.append(root_dir)

print("Root dir:", root_dir)

# 导入项目里的模块，保证 torch.load 反序列化时能找到类定义
import core.foundation_stereo
import core.extractor
import core.update
import core.submodule

# 权重路径
model_path = os.path.join(root_dir, "weights", "23-36-37", "model_best_bp2_serialize.pth")
print("Model path:", model_path)

# 加载整个模型
model = torch.load(model_path, map_location="cpu", weights_only=False)

print("=" * 80)
print("Model type:", type(model))
print("=" * 80)

# 打印前 100 个参数名
for i, (name, param) in enumerate(model.named_parameters()):
    print(f"{name:80s} | shape={tuple(param.shape)} | requires_grad={param.requires_grad}")
    if i >= 99:
        print("... only showing first 100 parameters")
        break