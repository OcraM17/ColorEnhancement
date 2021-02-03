from __future__ import print_function
import os
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.parallel,torch.nn.functional as functional
import torch.optim as optim
import torch.utils.data
import torchvision
import collections
from utility.ptcolor import rgb2lab
import model as model
from utility import data as data, Relu_Softmax_LAB as softquant,Qnt
from RGB_LAB_LCH_Colorization.trainmodel import cycle,save_model
from RGB_LAB_LCH_Colorization.trainmodel import initialization
from RGB_LAB_LCH_Colorization.Lab_Loss import Hist_2_Dist_AB,Hist_2_Dist_L
import signal
import tensorboardX
from Gan_Colorization.Discriminator_Hist import D_Hist


def main(args):
    criterionD = nn.BCELoss()
    criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0
    resnet = torchvision.models.resnet34(pretrained=True)
    resnet=resnet.to(args['device'])
    Generator = model.create_net(args['basis'], args['param'])
    #Generator.to(args['device'])
    if args['tab']:
        D = D_Hist(args['livab'],args['livL'],int(args['image_size']/4))
        tabL=Qnt.quantL(args['livL'],args['max_L'],args['min_L'])
        tabAB=Qnt.quantAB(args['livab'],args['min_max_ab'],-args['min_max_ab'])
    else:
        D = D_Hist(args['livab'], args['livL'])
    D.to(args['device'])
    optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(Generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    dataset = data.FiveKDataset(args['training_list'], args['raw_dir'],
                                args['expert_dir'], True, args['image_size'])
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args['batch_size'],
                                         shuffle=True,
                                         num_workers=args['num_workers'])
    model_pathG = os.path.join(args['output_dir'], "modelG")
    model_pathD = os.path.join(args['output_dir'], "modelD")
    step = initialization(model_pathG, Generator, optimizerG)
    step=initialization(model_pathD,D,optimizerD)
    interrupted = False
    def handler(sig, frame):
        nonlocal interrupted
        interrupted = interrupted or print("Training interrupted") or True

    signal.signal(signal.SIGINT, handler)

    writer= tensorboardX.SummaryWriter(args['output_dir'])
    ErrDRe_loss_history = collections.deque(maxlen=100)
    ErrD_loss_history = collections.deque(maxlen=100)
    ErrG_history=collections.deque(maxlen=100)
    data_iter = cycle(loader)

    for raw_cpu, expert_cpu in data_iter:
        if step >= args['iterations'] or interrupted:
            break
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        D.zero_grad()
        # Format batch
        x=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(expert_cpu)
        vector=resnet(x).detach()
        #vector=vector.view(-1)
        real_cpu = rgb2lab(expert_cpu.to(args['device']))
        real_cpu_down=functional.avg_pool2d(real_cpu,4)
        if args['tab']:
            ab=Hist_2_Dist_AB(real_cpu_down[:,1:2,:,:],tabAB,args['alpha'])
            L=Hist_2_Dist_L(real_cpu_down[:,0,:,:],tabL,args['alpha'])
        else:
            L = softquant.softhist_L(real_cpu_down[:, 0, :, :], args['min_L'], args['max_L'], args['livL']).to(args['device'])
            ab = softquant.softhist_AB(real_cpu_down, args['min_max_ab'], args['livab']).to(args['device'])


        b_size = real_cpu.size(0)

        label = torch.full((b_size,), real_label, device=args['device'],dtype=torch.float32)
        # Forward pass real batch through D
        output = D(ab,L,vector.to(args['device'])).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterionD(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch

        fake = Generator(raw_cpu.to(args['device']))
        label.fill_(fake_label)
        norm = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        fake = fake / torch.clamp(fake.max(axis=1, keepdim=True)[0], min=1)
        x=norm(fake)
        vector=resnet(x).detach()
        fake_down=functional.avg_pool2d(fake,4) #To avoid out of memory
        fake_down=rgb2lab(fake_down)
        if args['tab']:
            ab=Hist_2_Dist_AB(fake_down[:,1:2,:,:],tabAB,args['alpha'])
            L=Hist_2_Dist_L(fake_down[:,0,:,:],tabL,args['alpha'])
        else:
            L = softquant.softhist_L(fake_down[:, 0, :, :], args['min_L'], args['max_L'], args['livL']).to(args['device'])
            ab = softquant.softhist_AB(fake_down, args['min_max_ab'], args['livab']).to(args['device'])

        # Classify all fake batch with D
        output = D(ab.detach(),L.detach(),vector.to(args['device'])).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterionD(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        Generator.zero_grad()
        # rifre fke = ...
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = D(ab,L,vector.to(args['device'])).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        if step % 100 == 0:
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (
            step, args['iterations'], errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        ErrDRe_loss_history.append(errD_real.item())
        ErrD_loss_history.append(errD.item())
        ErrG_history.append(errG.item())

        writer.add_scalar("lOSS D RE", sum(ErrDRe_loss_history) / max(1, len(ErrDRe_loss_history)), step)
        writer.add_scalar("lOSS D ", sum(ErrD_loss_history) / max(1, len(ErrD_loss_history)), step)
        writer.add_scalar("lOSS G ", sum(ErrG_history) / max(1, len(ErrG_history)), step)
        step += 1
        if step % 5000 == 0:
            fake_sum = torchvision.utils.make_grid(torch.clamp(fake, 0, 1), nrow=8)
            writer.add_image("fake", fake_sum, step)
            ab_sum = torchvision.utils.make_grid(ab.unsqueeze(1).cpu(), nrow=8)
            writer.add_image("ab", ab_sum, step)            
        if step % args['save_every'] == 0 or step == args['iterations'] or interrupted:
            save_model(model_pathG + "_" + str(step) + ".pt", Generator, optimizerG, step, args)
            save_model(model_pathD+ "_" + str(step) + ".pt", D, optimizerD, step, args)
    data_iter.close()
    print("Training completed")
    # Output training stats


if __name__ == "__main__":
    DEFAULT_TRAINING_LIST = "/Users/marco/PycharmProjects/fivek/train1+2-list.txt"
    DEFAULT_RAW_DIR = "/Users/marco/PycharmProjects/fivek/raw"
    DEFAULT_EXPERT_DIR = "/Users/marco/PycharmProjects/fivek/expC"
    DEFAULT_VAL_LIST = "/Users/marco/PycharmProjects/fivek/test-list.txt"
    DEFAULT_OUTPUT_DIR = "/Users/marco/PycharmProjects/fivek/output"
    MODEL_PATH = '../'
    args = {
        'basis': 'splines',
        'param': 10,
        'output_dir': DEFAULT_OUTPUT_DIR,
        'training_list': DEFAULT_TRAINING_LIST,
        'raw_dir': DEFAULT_RAW_DIR,
        'expert_dir': DEFAULT_EXPERT_DIR,
        'image_size': 256,
        'iterations': 150000,
        'save_every': 25000,
        'start_from': None,
        'validation_list': DEFAULT_VAL_LIST,
        'validate_every': 1000,
        'batch_size': 4,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
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
