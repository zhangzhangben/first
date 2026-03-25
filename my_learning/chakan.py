from PIL import Image
import numpy as np

disp = np.array(Image.open("/Users/zhangyaben/Downloads/manipulation_v5_realistic_kitchen_2500_1/dataset/data/left/disparity/0000.png"))
print("shape:", disp.shape)
print("dtype:", disp.dtype)
print("min:", disp.min())
print("max:", disp.max())
print("unique sample:", np.unique(disp)[:20])