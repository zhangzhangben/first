from PIL import Image
import numpy as np

def depth_uint8_decoding(depth_uint8, scale=1000):
    depth_uint8 = depth_uint8.astype(float)
    out = depth_uint8[...,0]*255*255 + depth_uint8[...,1]*255 + depth_uint8[...,2]
    return out/float(scale)

img = np.array(Image.open("/Users/zhangyaben/Downloads/manipulation_v5_realistic_kitchen_2500_1/dataset/data/left/disparity/0000.png"))
disp = depth_uint8_decoding(img)

print("decoded shape:", disp.shape)
print("decoded dtype:", disp.dtype)
print("min:", disp.min())
print("max:", disp.max())
print("sample:", disp[0,0], disp[100,100], disp[300,300])