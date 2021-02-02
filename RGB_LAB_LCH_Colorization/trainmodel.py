#!/usr/bin/env python3
import torch
from torch.utils import data
import torchvision
import tensorboardX
import argparse
import os
import signal
import collections
import model as model
from utility import plots as plots, ptcolor as ptcolor, ptutils as ptutils, data as data
from RGB_LAB_LCH_Colorization import Lab_Loss,LCH_Loss,RBG_Loss

"""
Training Module
"""

def cycle(seq):
    while True:
        for x in seq:
            yield x

"""
save model: save the current model.
    parameters: 
        -path: path where the model's weights could be saved
        -optimizer: instance of the optimizer
        -step: the actual step
        -args: argument parsed 
"""
def save_model(path, model, optimizer, step, args):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": args,
    }, path + ".temp")
    os.rename(path+".temp", path)  # Replace atomically

"""
initialization: this method tries to load the previously saved weights of the model and recover a previously interrupted training.
                otherwise the main program will start a complete new training
    parameters:
        -path: path of the weights to be loaded
        -model: instance of the model
        -optimizer: instance of the optimizer
    
"""
def initialization(path, model, optimizer):
    try:
        data = torch.load(path)
    except FileNotFoundError:
        print("Starting from step 1")
        return 0
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    print("Continue from step", data["step"])
    return data["step"]

"""
compute error: compute the DeltaE error
    parameters:
        -x: image
        -y: target
"""
def compute_error(x, y):
    labx = ptcolor.rgb2lab(x)
    laby = ptcolor.rgb2lab(y)
    de = ptcolor.deltaE(labx, laby)
    return de

"""
make_summary_image: method called during validation. It generates an image using the actual snapshot of the net.
"""
def make_summary_image(model, args):
    dataset = data.FiveKDataset(args['validation_list'], args['raw_dir'],
                                args['expert_dir'], False, args['image_size'])
    loader = data.DataLoader(dataset,
                                         batch_size=args['batch_size'],
                                         shuffle=True)
    for pair in loader:
        raw = pair[0].to(args['device'])
        expert = pair[1].to(args['device'])
        break
    im = plots.make_test_image(32, raw.size(0))
    model.eval()
    with torch.no_grad():
        enhanced = model(raw)
        im = model(raw, raw.new_tensor(im)).cpu().numpy()
    model.train()
    pls = plots.plots_from_test_image(im, raw.size(2), raw.size(3))
    pls = raw.new_tensor(pls)
    sum_image = torch.cat([raw, pls, enhanced, expert], 3)
    sum_image = torchvision.utils.make_grid(sum_image, 2)
    return sum_image

"""
validate_model: it computes the validation error given the actual snapshot of the net.
"""
def validate_model(model, args):
    dataset = data.FiveKDataset(args['validation_list'], args['raw_dir'],
                                args['expert_dir'], False, args['image_size'])
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args['batch_size'],
                                         shuffle=False,
                                         num_workers=args['num_workers'])
    errors=[]
    model.eval()
    with torch.no_grad():
        for raw_cpu, expert_cpu in loader:
            raw = raw_cpu.to(args['device'])
            expert = expert_cpu.to(args['device'])
            enhanced = model(raw)
            if args['Color_Space'] == 'RGB':
                loss = RBG_Loss.loss_Quantization_RGB(enhanced, expert, args['alpha'], args['levels'])
            elif args['Color_Space'] == 'LAB':
                (loss, loss_L, loss_ab) = Lab_Loss.Lab_loss(enhanced, expert, args['alpha'], args['weight'],
                                                            args['levels'], args['min_ab'], args['max_ab'])
            elif args['Color_Space'] == 'LCH':
                (loss, loss_L, loss_C, loss_H) = LCH_Loss.lch_Loss(enhanced, expert, args['weight_C'],
                                                                   args['weight_H'], args['H_levels'], args['eps'])
            else:
                loss = torch.mean(compute_error(expert, enhanced))
            errors.append(loss.item())
    model.train()
    return sum(errors) / max(1, len(errors))




# training
def main(args):
    #Net
    net = model.create_net(args['basis'], args['param'])
    #net.to(args['device'])

    if args['Color_Space']=='LAB':
        train_lossL_history = collections.deque(maxlen=100)
        train_lossAB_history=collections.deque(maxlen=100)
    elif args['Color_Space']=='LCH':
        train_lossL_history = collections.deque(maxlen=100)
        train_lossC_history = collections.deque(maxlen=100)
        train_lossH_history = collections.deque(maxlen=100)
    train_loss_history = collections.deque(maxlen=100)


    #Optimizers and dataLoaders

    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=args['learning_rate'],
                                 weight_decay=args['weight_decay'])

    dataset = data.FiveKDataset(args['training_list'], args['raw_dir'],
                                args['expert_dir'], True, args['image_size'])

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args['batch_size'],
                                         shuffle=True,
                                         num_workers=args['num_workers'])


    model_path = os.path.join(args['output_dir'], "model.pt")
    step = initialization(model_path, net, optimizer)
    if args['start_from'] is not None:
        net.load_state_dict(torch.load(args['start_from'])["model"])


    writer = tensorboardX.SummaryWriter(args['output_dir'])
    writer.add_text("Options", str(args), step)
    display = ptutils.Display("   ".join(["Step {step}", "loss {loss:.5f}",
                                          "speed {steps_s:.2f} steps/s"]))
    display_validation = ptutils.Display("STEP {step}   VALIDATION ERROR {valid_err:.4f}")


    interrupted = False
    def handler(sig, frame):
        nonlocal interrupted
        interrupted = interrupted or print("Training interrupted") or True
    signal.signal(signal.SIGINT, handler)



    data_iter = cycle(loader)
    for raw_cpu, expert_cpu in data_iter:
        if step >= args['iterations'] or interrupted:
            break
        raw = raw_cpu.to(args['device'])
        expert = expert_cpu.to(args['device'])
        optimizer.zero_grad()
        enhanced = net(raw)

        if args['Color_Space'] == 'RGB':
            loss = RBG_Loss.loss_Quantization_RGB(enhanced, expert, args['alpha'], args['levels'])
        elif args['Color_Space']=='LAB':
            (loss,loss_L,loss_ab)=Lab_Loss.Lab_loss(enhanced,expert,args['alpha'],args['weight'],args['levels'],args['min_ab'],args['max_ab'])
            train_lossL_history.append(loss_L.item())
            train_lossAB_history.append(loss_ab.item())
        elif args['Color_Space']=='LCH':
            (loss, loss_L, loss_C, loss_H) = LCH_Loss.lch_Loss(enhanced, expert, args['weight_C'] ,args['weight_H'],args['H_levels'], args['eps'])
            train_lossL_history.append(loss_L.item())
            train_lossC_history.append(loss_C.item())
            train_lossH_history.append(loss_H.item())
        else:
            loss = torch.mean(compute_error(expert, enhanced))

        train_loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
        step += 1

        if step % 100 == 0:
            mean_loss = sum(train_loss_history) / max(1, len(train_loss_history))
            writer.add_scalar("loss", mean_loss, step)
            if args['Color_Space'] == 'LAB':
                mean_lossL = sum(train_lossL_history) / max(1, len(train_lossL_history))
                mean_lossAB = sum(train_lossL_history) / max(1, len(train_lossL_history))
                writer.add_scalar("loss_L", mean_lossL, step)
                writer.add_scalar("loss_AB", mean_lossAB, step)
            elif args['Color_Space']=='LCH':
                mean_lossL = sum(train_lossL_history) / max(1, len(train_lossL_history))
                mean_lossC = sum(train_lossC_history) / max(1, len(train_lossC_history))
                mean_lossH = sum(train_lossH_history) / max(1, len(train_lossH_history))
                writer.add_scalar("loss_L", mean_lossL, step)
                writer.add_scalar("loss_C", mean_lossC, step)
                writer.add_scalar("loss_H", mean_lossH, step)

            display.disp(step, loss=mean_loss)
        if step % args['validate_every'] == 0 and args['validation_list ']is not None:
            valid_err = validate_model(net, args)
            writer.add_scalar("validation_error", valid_err, step)
            display_validation.disp(step, valid_err=valid_err)
            sum_image = make_summary_image(net, args)
            writer.add_image("display", sum_image.cpu(), step)
        if step % args['save_every'] == 0 or step == args['iterations'] or interrupted:
            save_model(model_path, net, optimizer, step, args)
            writer.add_text("Save", "Model saved at step {}\n".format(step), step)
    data_iter.close()
    print("Training completed")


if __name__ == "__main__":
    DEFAULT_TRAINING_LIST = "/Users/marco/PycharmProjects/fivek/train1+2-list.txt"
    DEFAULT_RAW_DIR = "/Users/marco/PycharmProjects/fivek/raw"
    DEFAULT_EXPERT_DIR = "/Users/marco/PycharmProjects/fivek/expC"
    DEFAULT_VAL_LIST = "/Users/marco/PycharmProjects/fivek/test-list.txt"
    DEFAULT_OUTPUT_DIR="/Users/marco/PycharmProjects/fivek/output"
    MODEL_PATH='../'
    args={
        'basis':'splines',
        'param':10,
        'output_dir':DEFAULT_OUTPUT_DIR,
        'training_list':DEFAULT_TRAINING_LIST,
        'raw_dir':DEFAULT_RAW_DIR,
        'expert_dir': DEFAULT_EXPERT_DIR,
        'image_size':256,
        'iterations':150000,
        'save_every':25000,
        'start_from':None,
        'validation_list':DEFAULT_VAL_LIST,
        'validate_every':1000,
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
