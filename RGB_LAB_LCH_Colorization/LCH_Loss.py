import torch
from utility import ptcolor as ptcolor

"""
LCH Loss:
This file compute the Loss in the LCH color space: |L-LT|+|C-CT|+crossentropy(H,HT)
the method hue to distribution convert the H channel in a probability distribution (via quantization).
"""

def lch_Loss(img,gt,weightC,weightH,levels,eps=0.01,weight=None):
    img_lch= ptcolor.rgb2lch(img)
    gt_lch= ptcolor.rgb2lch(gt)
    loss_L=torch.mean(torch.abs(img_lch[:,0,:,:]-gt_lch[:,0,:,:]))
    loss_C=torch.mean(torch.abs(img_lch[:,1,:,:]-gt_lch[:,1,:,:]))
    img_H_Dist=torch.clamp(hue_to_distribution(img_lch[:,2,:,:],levels,eps),0.001, 0.999)
    gt_H_Dist =torch.clamp(hue_to_distribution(gt_lch[:, 2, :, :], levels),0.001, 0.999)
    if weight is None:
        loss_H = torch.mean(-torch.mul(gt_H_Dist, torch.log(img_H_Dist)))
    else:
        loss_H = -(gt_lch[:,1,:,:]*(gt_H_Dist*torch.log(img_H_Dist)).sum(1,keepdim=True)).mean()
    loss=loss_L+weightC*loss_C+weightH*loss_H
    return(loss,loss_L,loss_C,loss_H)

def hue_to_distribution(h, levels, eps=0.0):
    h = h * (levels / 360.0)
    a = torch.arange(levels).float().to(h.device)
    a = a.view(1, levels, 1, 1)
    h=h.unsqueeze(1)
    p = torch.relu(1 - torch.abs(h - a))
    p = p + (a == 0.0).float() * p[:, -1:, :, :]
    p = (p + torch.ones_like(p) * eps) / (1.0 + levels * eps)
    return p

if __name__=="__main__":
    h=torch.ones(4,3,16,16)
    g=torch.ones(4,3,16,16)
    levels=4
    print(lch_Loss(h,g,1,1,levels,0.01,True)[0])
