from utility.ptcolor import rgb2lab
from utility.Qnt import quantAB
import torch

"""
Lab_Loss: These methods compute the crossentropy loss of the distribution in the LAB color space.
The methods Hist_2_dist convert the difference between the image and the table (i.e. the quantization) in distribution.
"""
def Hist_2_Dist_L(img, tab):
    img_dist=((img.unsqueeze(1)-tab)**2)
    p=torch.nn.functional.softmax(img_dist,dim=1)
    return p

def Hist_2_Dist_AB(img,tab,levels,alpha):
    img_dist=((img.unsqueeze(1)-tab)**2).sum(2)
    p = torch.nn.functional.softmax(img_dist, dim=1)
    return p

def loss_ab(img,gt,alpha,tab,levels):
    p= Hist_2_Dist_AB(img, tab, levels,alpha)
    q= Hist_2_Dist_AB(gt,tab, levels, alpha,)
    p = torch.clamp(p, 0.001, 0.999)
    loss = -(q*torch.log(p)).sum([1,2,3]).mean()
    return loss

def Lab_loss(img,gt,alpha,weight,levels,vmin,vmax):
    tab=quantAB(levels,vmin,vmax)
    lab_img=torch.clamp(rgb2lab(img),vmin,vmax)
    lab_gt=torch.clamp(rgb2lab(gt),vmin,vmax)

    loss_l=torch.abs(lab_img[:,0,:,:]-lab_gt[:,0,:,:]).mean()
    loss_AB=loss_ab(lab_img[:,1:,:,:],lab_gt[:,1:,:,:],alpha,tab,levels)
    loss=loss_l+weight*loss_AB
    return (loss,loss_l,loss_AB)



if __name__ =="__main__":
    img=torch.ones(1,3,10,10)
    gt=torch.ones(1,3,10,10)
    print(Lab_loss(img,gt,1,1,3,-80,80))
