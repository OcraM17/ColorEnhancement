import torch
import lpips
"""
PSNR: images should be tensors with values between 0 and 1
"""
def PSNR(img,gt):
    mseL=torch.nn.MSELoss()
    mse=mseL(img,gt)
    return 20*torch.log10(1/torch.sqrt(mse))

def Lpips(img,gt):
    lpLoss=lpips.LPIPS(pretrained=True,net='alex')
    return lpLoss(img,gt).item()

if __name__=='__main__':
    a=torch.ones([1,3,64,64])
    b=torch.ones([1,3,64,64])
    print(Lpips(a,b))