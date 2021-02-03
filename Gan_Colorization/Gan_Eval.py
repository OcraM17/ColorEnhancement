import torch
import argparse
import os
import model as model
from utility import plots as plots, data as data, Relu_Softmax_LAB
import numpy as np
import imageio
from Gan_Colorization.Discriminator_Hist import D_Hist
from torchvision.models import resnet34
from torchvision.transforms import Normalize
import utility.Relu_Softmax_LAB as softquant
from utility import Qnt
from RGB_LAB_LCH_Colorization.Lab_Loss import Hist_2_Dist_AB,Hist_2_Dist_L
"""
Evaluation Module for the GAN project:
"""

def main(args):
    resnet=resnet34(pretrained=True)
    resnet.to(args['device'])
    ldataG = torch.load(os.path.join(args['output_dir'],args['model_G']))
    ldataD = torch.load(os.path.join(args['output_dir'],args['model_D']))
    basis = ldataG["args"]['basis']
    param = ldataG["args"]['param']
    net = model.create_net(basis, param)
    net.load_state_dict(ldataG["model"])
    #net.to(args['device'])

    if args['tab']:
        D = D_Hist(args['livab'], args['livL'], int(args['image_size'] / 4))
        tabL = Qnt.quantL(args['livL'], args['max_L'], args['min_L'])
        tabAB = Qnt.quantAB(args['livab'], args['min_max_ab'], -args['min_max_ab'])
    else:
        D = D_Hist(args['livab'], args['livL'])

    D.load_state_dict(ldataD["model"])
    D.to(args['device'])

    dataset = data.FiveKDataset(args['test_list'], args['raw_dir'],
                                args['expert_dir'], False, args['image_size'],
                                filenames=True)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args['batch_size'],
                                         shuffle=False,
                                         num_workers=args['num_workers'])

    net.eval()
    with torch.no_grad():
        for raw_cpu, expert_cpu, filename in loader:
            raw = raw_cpu.to(args['device'])
            enhanced = net(raw)
            enhanced = enhanced / torch.clamp(enhanced.max(axis=1, keepdim=True)[0], min=1)
            norm=Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            x=norm(enhanced)
            vector=resnet(x)
            enhanced_D=torch.nn.functional.avg_pool2d(enhanced,4)
            if args['tab']:
                ab = Hist_2_Dist_AB(enhanced_D[:, 1:2, :, :], tabAB, args['alpha'])
                L = Hist_2_Dist_L(enhanced_D[:, 0, :, :], tabL, args['alpha'])
            else:
                L = softquant.softhist_L(enhanced_D[:, 0, :, :], args['min_L'], args['max_L'], args['livL']).to(args['device'])
                ab = softquant.softhist_AB(enhanced_D, args['min_max_ab'], args['livab']).to(args['device'])
            D(ab,L,vector.to(args['device']))
            if args['output_dir'] is not None:
                test_im = plots.make_test_image(32, raw.size(0))
                test_ims = net(raw, raw.new_tensor(test_im))
                test_ims = test_ims.cpu().numpy()
                pls = plots.plots_from_test_image(test_ims, raw.size(2), raw.size(3), True)
                pls = (pls.transpose([0, 2, 3, 1]) * 255).astype(np.uint8)
                ims = torch.clamp(enhanced, 0, 1).cpu().numpy()
                ims = (ims.transpose([0, 2, 3, 1]) * 255).astype(np.uint8)
                for i in range(ims.shape[0]):
                    p = os.path.join(args['output_dir'], filename[i])
                    imageio.imsave(p, ims[i])
                    p = os.path.join(args['output_dir'], filename[i].replace(".", "-plot.", 1))
                    imageio.imsave(p, pls[i])

if __name__=="__main__":

    DEFAULT_TEST_LIST = "/Users/marco/PycharmProjects/fivek/train1+2-list.txt"
    DEFAULT_RAW_DIR = "/Users/marco/PycharmProjects/fivek/raw"
    DEFAULT_EXPERT_DIR = "/Users/marco/PycharmProjects/fivek/expC"
    DEFAULT_OUTPUT_DIR = "/Users/marco/PycharmProjects/fivek/output"
    MODEL_PATHG = 'modelG_3.pt'
    MODEL_PATHD = 'modelD_3.pt'
    args = {
        'model_G':MODEL_PATHG,
        'model_D':MODEL_PATHD,
        'basis': 'splines',
        'param': 10,
        'output_dir': DEFAULT_OUTPUT_DIR,
        'test_list': DEFAULT_TEST_LIST,
        'raw_dir': DEFAULT_RAW_DIR,
        'expert_dir': DEFAULT_EXPERT_DIR,
        'image_size': 256,
        'batch_size': 4,
        'num_workers': 0,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'min_L':0,
        'max_L':100,
        'min_max_ab':80,
        'weight':1,
        'livL':50,
        'livab':20,
        'alpha':1,
        'tab':True
    }
    main(args)
