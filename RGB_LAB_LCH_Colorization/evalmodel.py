
import torch
import argparse
import os
import model
from RGB_LAB_LCH_Colorization import LCH_Loss,RBG_Loss,Lab_Loss
from utility import plots, ptcolor, data
import numpy as np
import imageio
"""
evalmodel: evaluation module
"""
def compute_errors(x, y):
    """
    compute errors: It computes the deltae,deltae94 and absoulte difference on l channel
        params:
            -x: input image [B,C,H,W]
            -y: target image [B,C,H,W]
        return:
            -de,de94,dl: deltae,deltae94 and absoulte difference on l channel
    """
    labx = ptcolor.rgb2lab(x)
    laby = ptcolor.rgb2lab(y)
    de = ptcolor.deltaE(labx, laby)
    de94 = ptcolor.deltaE94(labx, laby)
    dl = torch.abs(labx[:, 0, :, :] - laby[:, 0, :, :])
    return de, de94, dl


def main(args):
    ldata = torch.load(os.path.join(args['output_dir'], "model.pt"))
    basis = ldata["args"]['basis']
    param = ldata["args"]['param']
    net = model.create_net(basis, param)
    net.load_state_dict(ldata["model"])
    #net.to(args['device'])
    dataset = data.FiveKDataset(args['test_list'], args['raw_dir'],
                                args['expert_dir'], False, args['image_size'],
                                filenames=True)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args['batch_size'],
                                         shuffle=False,
                                         num_workers=args['num_workers'])

    net.eval()
    l=[]
    with torch.no_grad():
        for raw_cpu, expert_cpu, filename in loader:
            raw = raw_cpu.to(args['device'])
            expert = expert_cpu.to(args['device'])
            enhanced = net(raw)
            if args['Color_Space'] == 'RGB':
                loss = RBG_Loss.loss_Quantization_RGB(enhanced, expert, args['alpha'], args['levels'])
            elif args['Color_Space'] == 'LAB':
                (loss, loss_L, loss_ab) = Lab_Loss.Lab_loss(enhanced, expert, args['alpha'], args['weight'],
                                                            args['levels'], args['min_ab'], args['max_ab'])
            elif args['Color_Space'] == 'LCH':
                (loss, loss_L, loss_C, loss_H) = LCH_Loss.lch_Loss(enhanced, expert, args['weight_C'], args['weight_H'],
                                                                   args['H_levels'], args['eps'])
            else:
                loss, de94, dl = compute_errors(expert, enhanced)

            l.append(loss)

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

    print("{:.2f}".format(sum(l)/len(l)))

if __name__ == "__main__":
    DEFAULT_RAW_DIR = "/Users/marco/PycharmProjects/fivek/raw"
    DEFAULT_EXPERT_DIR = "/Users/marco/PycharmProjects/fivek/expC"
    DEFAULT_TEST_LIST = "/Users/marco/PycharmProjects/fivek/test-list.txt"
    DEFAULT_OUTPUT_DIR="/Users/marco/PycharmProjects/fivek/output"
    MODEL_PATH='../'
    args={
        'basis':'splines',
        'param':10,
        'output_dir':DEFAULT_OUTPUT_DIR,
        'test_list':DEFAULT_TEST_LIST,
        'raw_dir':DEFAULT_RAW_DIR,
        'expert_dir': DEFAULT_EXPERT_DIR,
        'image_size':256,
        'save_every':25000,
        'start_from':None,
        'batch_size':4,
        'learning_rate':1e-3,
        'weight_decay':1e-5,
        'num_workers':0,
        'device':torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        #Color space could assume a value among None,RGB,LAB,LCH
        'Color_Space':'LCH'
    }

    if args['Color_Space']=='RGB' or args['Color_Space']=='LAB':
        args['levels']=7
        args['alpha']=1

    if args['Color_Space']=='LAB':
        args['max_ab']=80
        args['min_ab']=-80
        args['weight']=1

    if args['Color_Space']=='LCH':
        args['weight_C']=1
        args['weight_H']=1
        args['H_levels']=4
        args['eps']=0.01

    main(args)