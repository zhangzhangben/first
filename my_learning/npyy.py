import numpy as np
a = np.load('output1/disp.npy')
b = np.load('/root/autodl-tmp/Fast-FoundationStereo-master/exp_refine_clean/split/Hard_Test_100/preds_npy/0040.npy')
print('shape a:', a.shape)
print('shape b:', b.shape)
print('mean abs diff:', np.abs(a - b).mean())
print('max abs diff:', np.abs(a - b).max())
