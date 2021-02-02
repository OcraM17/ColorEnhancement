import torch
from utility.Qnt import quantRGB
"""
RGB_Loss:
This collection of methods compute the RGB crossentropy loss between the distributions.
the method hist_2_Dist conmverts the histogram in distribution.
"""
def Hist_2_Dist_RGB(img, tab, levels, alpha=1):
    img_dist = ((img.unsqueeze(1)-tab)**2).sum(2)
    p = torch.nn.functional.softmax(torch.mul(-alpha, img_dist/(levels**3)),dim=1)
    return p

def loss_Quantization_RGB(img, gt, alpha, levels):
    tab=quantRGB(levels)
    p = Hist_2_Dist_RGB(img,tab, levels,alpha)
    q = Hist_2_Dist_RGB(gt, tab,levels, alpha)
    p = torch.clamp(p, 0.001, 0.999)
    loss = -(q*torch.log(p)).sum([1,2,3]).mean()
    return loss

if __name__ == "__main__":
    x = torch.ones(4, 3, 64, 64)
    y= torch.zeros(4,3,64,64)
    print(loss_Quantization_RGB(x,y,1,2))