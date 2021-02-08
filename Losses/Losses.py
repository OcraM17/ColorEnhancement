import torch
import lpips
"""
PSNR: images should be tensors with values between 0 and 1
"""
def PSNR(img,gt):
    mseL=torch.nn.MSELoss()
    mse=mseL(img,gt)
    return 20*torch.log10(1/torch.sqrt(mse))

if __name__=='__main__':
    a=torch.ones([1,3,64,64])
    b=torch.ones([1,3,64,64])
    print(PSNR(a,b))
    l=lpips.LPIPS(pretrained=True,net='alex')
    print(l(a,b).item())
