import torch
from torch.nn import BCELoss
from torch.nn import functional as F
"""
Discriminator Histogram:
    It takes in input the L,AB histograms of the batch and the resnet's output vector.
    constructor paramters:
        -livab: number of the bins for A and B
        -livL: number of the bins for L
        -img_size: size of the input image. if size is None, it waits directly in input the histogram of the entire image [B,Lev](Relu softmax), otherwise, [B,Lev,H,W](softmax)
    
    forward parameters:
        -img_ab: histogram for channel A and B of the image
        -img_L: histogram for channel L of the image
        -vect: resnet's features output
    
    the output of the discriminator is the probability of the image to be classified as fake or real.
"""

class D_Hist(torch.nn.Module):
    def __init__(self,livab,livL,img_size=None):
        super().__init__()
        self.img_size=img_size
        if self.img_size is not None:
            self.L=torch.nn.Linear(img_size*img_size*livL,128)
            self.ab = torch.nn.Linear(img_size * img_size * livab * livab, 128)
        else:
            self.ab = torch.nn.Linear(livab * livab, 128)
            self.L = torch.nn.Linear(livL, 128)

        self.vector=torch.nn.Linear(1000,128)
        self.l2=torch.nn.Linear(128,64)
        self.l3=torch.nn.Linear(64,32)
        self.l4=torch.nn.Linear(32,1)
        self.fin=torch.nn.Sigmoid()
    def forward(self,img_ab,img_l,vect):
        if self.img_size is not None:
            ab=img_ab.view(-1,img_ab.size(1)*img_ab.size(2)*img_ab.size(3))
            L = img_l.view(-1, img_l.size(1) * img_l.size(2) * img_l.size(3))
        else:
            ab = img_ab.view(-1, img_ab.size(1) * img_ab.size(2))
            L = img_l.view(-1, img_l.size(1))

        ab=F.leaky_relu(self.ab(ab))
        L=F.leaky_relu(self.L(L))
        v=F.leaky_relu(self.vector(vect))

        x=(ab+L+v)

        x=F.leaky_relu(self.l2(x))
        x=F.leaky_relu(self.l3(x))
        x=self.l4(x)
        x=self.fin(x)
        return x


if __name__=="__main__":
    x=torch.ones(4,36,256,256)
    y=torch.ones(4,10,256,256)
    D=D_Hist(6,256,10)
    b_size = x.size(0)
    out=D(x,y).view(-1)
    # label = torch.full((b_size,), real_label, device=args.device)
    label = torch.full((b_size,), 1)
    # Forward pass real batch through D
    # Calculate loss on all-real batch
    criterion=BCELoss()
    errD_real = criterion(out, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    print(errD_real)
