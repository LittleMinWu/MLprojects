import numpy as np
def conv2d(input_array, kernels, stride=1, padding=0):
    H, W = input_array.shape
    kh, kw = kernels.shape
    p = padding
    out_h = (H + 2 * padding - kh) // stride + 1  # //floor division
    out_w = (W + 2 * padding - kw) // stride + 1  # //floor division
    outputs = np.zeros([  out_h, out_w])
    for h in range(out_h):
        for w in range(out_w):
            for x in range(kh):
                for y in range(kw):
                    outputs[h][w] += input_array[h * stride + x][w * stride + y] * kernels[x][y]
    return outputs
from PIL import Image
im=Image.open('a.jpg')
rgb=np.array(im.convert('RGB'))
r=rgb [ : , : , 0 ] # array of R pixels
Image.fromarray(np. uint8 ( r ) ).show ()
kernel1 = np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]
    ).reshape(3,3)
kernel2 = np.asarray(
        [
            [0, -1, 0],
            [-1, 8, -1],
            [0, -1, 0]
        ]
    )
output1=conv2d(r,kernel1)
output2=conv2d(r,kernel2)
print(output1)
Image.fromarray(np. uint8 ( output1 ) ).show ()
print(output2)
Image.fromarray(np. uint8 ( output2 ) ).show ()