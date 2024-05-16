import cv2
import numpy as np
import torch

from mslf import MSLF
from PIL import Image

#模型导入和边缘检测
def compare(network,checkpoint,image,name):
    model=network().cuda()
    c=torch.load(checkpoint)
    model.load_state_dict(c['state_dict'],strict=False)
    result=model(image)
    all_res = torch.zeros((len(result), 1, H, W))
    for i in range(len(result)):
        all_res[i, 0, :, :] = result[i]

    #torchvision.utils.save_image(1 - all_res, '11.jpg' )
    fuse_res = torch.squeeze(result[-1].detach()).cpu().numpy()
    fuse_res = ((1-fuse_res) * 255).astype(np.uint8)
    new_image=Image.fromarray(fuse_res)
    new_image.save("E:/Program/output/test2/0211.jpg")
    #print(fuse_res)


#图像预处理
image=cv2.imread('E:/Program/output/test2/021.jpg').astype(np.uint8)
#image=cv2.resize(image,(700,700))
image = torch.from_numpy(image)
image = torch.tensor(image, dtype=torch.float32)
image=image.unsqueeze(0)
image=image.permute(0,3,1,2)
image = image.cuda()
_, _, H, W = image.shape

compare(MSLF,"model/mslf1.pth",image,'mslf')

