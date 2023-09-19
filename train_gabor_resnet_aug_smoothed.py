import argparse

import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps
import PIL
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
# from torch.utils import tensorboard
import matplotlib.pyplot as plt
from ansim_dataset_unconf import ansimDataset_mix_smoothmask_s, ansimDataset_mix_smoothmask_s_randnoise
from ConvLSTM_unconf import MtConvLSTM
import random
import math
import torch.nn as nn
import torch.optim as optim
from SGLD import SGLD
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler

from model import GaborConv2dPC
from torchvision import models
from torchvision import transforms, utils
from model import ResidualBlock, ResNet, ResNet_gabor_cat2
from simmim import build_simmim


import time

torch.backends.cudnn.enabled=False

def train_model(args, model, smoothed_output_model, criterion, optimizer, scheduler, trainloader, testloader):
    since = time.time()

    # epoch_num = 60
    for epoch in range(1, args.iter+1):
        print('learning rate, ', optimizer.param_groups[0]["lr"])
        epoch_num = epoch + args.start_iter
        if epoch_num > args.iter:
            print('DONE TRAINING!')
            break

        print('Epoch {}/{}'.format(epoch_num, args.iter))
        print('-' * 10)

        # Each epoch has a training phase
        model.train(True)  # Set model to training mode
        running_loss = 0.0

        for inputs, targets, ss, mask, idx in trainloader:
            b, t, c, h, w = inputs.shape
            # print(inputs.shape)  #367, 367
            if t == 1:
                inputs = inputs.squeeze(1)
                targets = targets.squeeze(1)
                ss = ss.squeeze(1)

            else:
                inputs = inputs.reshape(b * t, c, h, w)
            # print('input', inputs.shape)

            if use_gpu:
                inputs, targets, ss, mask = inputs.cuda(), targets.cuda(), ss.cuda(), mask.cuda()
            else:
                inputs, targets = Variable(inputs), Variable(targets)

            # zero the parameter gradients
            optimizer.zero_grad()

            # use mixed precision
            with autocast():
                predicted = model(inputs)

                targets2 = targets * math.pi  # did not /255 since the target/input is already (0,1)
                targets_qxx = torch.pow(torch.cos(targets2), 2) - 1 / 2
                targets_qxy = torch.cos(targets2) * torch.sin(targets2)
                targets_q = torch.cat([targets_qxx, targets_qxy], dim = 1)

                #### smooth target
                # print('target q shape, ', targets_q.shape)
                smooth_loss, target_smooth = smoothed_output_model(targets_q, mask)

                predicted2 = predicted  * math.pi

                # apply s as a mask to the predicted2, so that the defect regions are not counted as loss, since it has no orientation
                predicted2 = predicted2 * ss
                target_smooth = target_smooth * ss

                predicted_qxx = torch.pow(torch.cos(predicted2), 2) - 1 / 2
                predicted_qxy = torch.cos(predicted2) * torch.sin(predicted2)
                predicted_q = torch.cat((predicted_qxx, predicted_qxy), dim=1)

                loss = criterion(predicted_q, target_smooth)

                # utils.save_image(
                #     predicted2,
                #     f"img_smooth_s/train_predicted2_{args.model_name}_{str(epoch_num).zfill(3)}.png", nrow=args.batch // 4)

            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            # statistics
            loss = loss*1000
            iter_loss = loss.item()
            running_loss += loss.item()
            epoch_loss = running_loss / len(trainset)

        print('{} Loss: {:.4f} batch_loss: {:f}'.format("train", epoch_loss, iter_loss))
        scheduler.step()

        with torch.no_grad():
            running_loss_test = 0.0
            loss_by_class = 0.0
            test_iter = 0
            # print('validating...')
            for test_inputs, test_targets, test_ss, mask, idx in testloader:
                b, t, c, h, w = test_inputs.shape
                if t == 1:
                    test_inputs = test_inputs.squeeze(1)
                    test_targets = test_targets.squeeze(1)
                    test_ss = test_ss.squeeze(1)
                else:
                    test_inputs = test_inputs.reshape(b * t, c, h, w)

                test_iter += 1

                if use_gpu:
                    test_inputs, test_targets, test_ss, mask = test_inputs.cuda(), test_targets.cuda(), test_ss.cuda(), mask.cuda()
                else:
                    test_inputs, test_targets = Variable(test_inputs), Variable(test_targets)


                test_predicted = model(test_inputs)

                test_targets2 = test_targets * math.pi  # / 255
                targets_qxx = torch.pow(torch.cos(test_targets2), 2) - 1 / 2
                targets_qxy = torch.cos(test_targets2) * torch.sin(test_targets2)
                targets_q = torch.cat((targets_qxx, targets_qxy), dim=1)

                #### smooth target
                # print('target q shape, ', targets_q.shape) #[16, 2, 320, 320]
                smooth_loss, target_smooth = smoothed_output_model(targets_q, mask)

                # print('after smooth, ', target_smooth.shape)
                # print('smooth loss, ', smooth_loss)

                test_predicted2 = test_predicted  * math.pi  #/ 255

                # apply s
                test_predicted2 = test_predicted2 * test_ss
                target_smooth = target_smooth * test_ss

                # print('predicted2, ss', test_predicted2.shape, test_ss.shape)

                predicted_qxx = torch.pow(torch.cos(test_predicted2), 2) - 1 / 2
                predicted_qxy = torch.cos(test_predicted2) * torch.sin(test_predicted2)
                predicted_q = torch.cat((predicted_qxx, predicted_qxy), dim=1)


                loss_test = criterion(predicted_q, target_smooth)

                loss_test = loss_test * 1000
                iter_loss_test = loss_test.item()
                running_loss_test += loss_test.item()
                epoch_loss_test = running_loss_test / len(testset)
                loss_by_class += loss_test.item()
                if test_iter == 20:
                    print('Loss on the 1-200: %.5f ' % (loss_by_class/200.0))
                    loss_by_class = 0.0
                elif test_iter == 40:
                    print('Loss on the 201-400: %.5f ' % (loss_by_class/200.0))
                    loss_by_class = 0.0
                elif test_iter == 50:
                    print('Loss on the 401-500: %.5f ' % (loss_by_class/100.0))
                    loss_by_class = 0.0
                elif test_iter == 70:
                    print('Loss on the 501-700: %.5f ' % (loss_by_class/200.0))
                    loss_by_class = 0.0
                elif test_iter == 80:
                    print('Loss on the 701-800: %.5f ' % (loss_by_class/100.0))
                    loss_by_class = 0.0
                epoch_loss_test = running_loss_test / len(testset)

            print('Loss on the test images: %.5f ' % (epoch_loss_test))

        if epoch_num % 50 == 0:
            # print('learning rate, ', optimizer.param_groups[0]["lr"])
            print('saving weights...')
            # output_path = "/work/yunruili/unconfined_orientation/experiment/gabor_resnet_epoch_%d.weights" % (epoch_num)
            torch.save({
                'model':model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
                output_path+args.model_name+".epoch.%d" % (epoch_num))

            print('saving images...')
            # for i in range(min(inputs.shape[0], test_inputs.shape[0])):
            utils.save_image(
                inputs,
                f"img_smooth_s/train_input_{args.model_name}_{str(epoch_num).zfill(3)}.png", nrow=args.batch//4)

            utils.save_image(
                targets,
                f"img_smooth_s/train_target_{args.model_name}_{str(epoch_num).zfill(3)}.png", nrow=args.batch//4)

            utils.save_image(
                predicted,
                f"img_smooth_s/train_pred_{args.model_name}_{str(epoch_num).zfill(3)}.png", nrow=args.batch//4)

            utils.save_image(
                test_inputs,
                f"img_smooth_s/test_input_{args.model_name}_{str(epoch_num).zfill(3)}.png", nrow=args.batch//4)

            utils.save_image(
                test_targets,
                f"img_smooth_s/test_target_{args.model_name}_{str(epoch_num).zfill(3)}.png", nrow=args.batch//4)

            utils.save_image(
                test_predicted,
                f"img_smooth_s/test_pred_{args.model_name}_{str(epoch_num).zfill(3)}.png", nrow=args.batch//4)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model

if __name__=='__main__':
    # img_path = '/home/rliu/ansim/data/unconfined_steph/cropped_orientation/'
    img_path = '/work/ruoshiliu/ansim/data/unconfined_steph/cropped_retardance/'
    img_path_input = '/work/ruoshiliu/ansim/data/unconfined_steph/cropped_retardance/'

    img_path_target = '/work/ruoshiliu/ansim/data/unconfined_steph/cropped_orientation/'
    # img_path_target = '/scratch0/yunruili/smooth_orientation_320/'   ### the saved smoothed image

    # img_path = '/home/yunruili/unconfined_orientation/data/'
    img_list_csv = 'data/img_list.csv'
    train_csv = 'data/train_unconf_random_36k.csv'
    test_csv = 'data/test_unconf_random_6k.csv'
    output_path = 'experiment/'

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("GPU in use")

    device = torch.cuda.device("cuda" if torch.cuda.is_available() else "cpu")

    print(torch.__version__)
    print(use_gpu, device)

    parser = argparse.ArgumentParser(description='gabor_resnet trainer')
    parser.add_argument('--model_name', type=str, default='gabor_resnet', help='name to save model weights')
    parser.add_argument('--aug', action='store_true', help="true for data augmentation in train loader")
    parser.add_argument("--iter", type=int, default=100, help="total training iterations")
    parser.add_argument("--batch", type=int, default=16, help="batch sizes")
    parser.add_argument("--step", type=int, default=2, help="total of input and forecast sequence")
    parser.add_argument("--image_size", type=int, default=128, help="image size")
    parser.add_argument("--rand_range", type=int, default=10, help="random range for the starting index in dataloader")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoint to resume training")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--lr_step", type=int, default=8, help="learning rate change after certain step")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="learning rate gamma for rate change")
    parser.add_argument("--train_gabor", type=str, default=True, help="fix gabor filter or not")
    parser.add_argument("--channels", type=str, default='64, 128, 256, 512', help="channels for the resnet model")
    parser.add_argument("--layers", type=str, default='2, 2, 2, 2', help="layers for the resnet model")
    parser.add_argument("--scale", type=int, default=8, help="scale for gabor filter")
    parser.add_argument("--orientation", type=int, default=8, help="orientation for gabor filter")
    parser.add_argument("--pc", type=bool, default=False, help="whether to calculate pc or not")
    parser.add_argument("--lam", type=str, default='2', help="wave length of gabor filter")
    parser.add_argument("--kernel", type=int, default=11, help="kernel size for gabor filter")
    # parser.add_argument("--kernel", type=str, default='11', help="kernel size of gabor filter")
    parser.add_argument("--cat", type=int, default=0, help="if set to 1, concat the original input to gabor output; set t0 2, 2 gabor filters; 3, 3 gabor filters")
    parser.add_argument("--att", type=int, default=0, help="if set to 1, use SE Layer for attention")
    args = parser.parse_args()

    args.start_iter = 0
    args.num_workers = 1
    # args.input_channel = args.scale * args.orientation * 2

    # batch_size = 8
    # step_size = 2  # 16
    # num_workers = 1
    # image_size = 128
    # num_epochs = 60
    # rand_range = 10

    # Iterate over data
    if args.aug:
        print('******* data augmented!')
        traindata = []
        for i in range(0, 360, 36):
            data = ansimDataset_mix_smoothmask_s_randnoise(img_list_csv=img_list_csv, seq_csv=train_csv,
                                                               root_dir_input=img_path_input,
                                                               root_dir_target=img_path_target, step=2, gap=1,
                                                               rotate_angel=i,
                                                               rotate=True, image_size=520, rand_range=0, noise_pct=0.1)
            # data = ansimDataset_mix_smoothmask_s(img_list_csv=img_list_csv, seq_csv=train_csv, root_dir_input=img_path_input,
            #                         root_dir_target=img_path_target, step=args.step, gap= 1, rotate_angel=i,
            #                         rotate=True, image_size=args.image_size, rand_range=args.rand_range)
            traindata.append(data)
        trainset = torch.utils.data.ConcatDataset(traindata)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True,
                                                  num_workers=args.num_workers)
    else:
        trainset = ansimDataset_mix_smoothmask_s_randnoise(img_list_csv=img_list_csv, seq_csv=train_csv,
                                                           root_dir_input=img_path_input,
                                                           root_dir_target=img_path_target, step=2, gap=1,
                                                           rotate_angel=0,
                                                           rotate=True, image_size=520, rand_range=0, noise_pct=0.1)
        # trainset = ansimDataset_mix_smoothmask_s(img_list_csv=img_list_csv, seq_csv=train_csv, root_dir_input=img_path_input,
        #                             root_dir_target=img_path_target, step=args.step, gap=1, rotate_angel=0,
        #                             rotate=True, image_size=args.image_size, rand_range=args.rand_range)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True,
                                                  num_workers=args.num_workers)

    print("trainloader ready!")
    print(len(trainset))

    testset = ansimDataset_mix_smoothmask_s(img_list_csv=img_list_csv, seq_csv=test_csv, root_dir_input=img_path_input,
                               root_dir_target=img_path_target, step=args.step, gap=1, rotate_angel=0,
                               rotate=True, image_size=args.image_size, rand_range=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    print("testloader ready!")
    print(len(testset))

    channels = args.channels.split(',')
    channels = [int(i) for i in channels]

    layers = args.layers.split(',')
    layers = [int(i) for i in layers]

    lam = args.lam.split(',')
    lam = [int(i) for i in lam]
    model = ResNet_gabor_cat2(ResidualBlock, layers, channels,
                     args.scale, args.orientation, args.kernel, args.pc, lam)

    print(model)
    count_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameter: %d" % count_param)


    if use_gpu:
        model = torch.nn.DataParallel(model)
        print('multiple GPUs!')
        model.cuda()

    criterion = nn.MSELoss()

    ##### build simmim smooth model
    simmim_path = '/work/yunruili/orientation_mask/'
    checkpoint_path = 'output/simmim_pretrain/orientation_pretrain__swin_base__img520_window4_mask10_ratio5__800ep_40k_FP32_crop320/ckpt_epoch_220.pth'


    print('************* building smooth model')
    smoothed_output_model = build_simmim()
    print('************* loading smooth model')
    if use_gpu:
        smoothed_output_model.cuda()
        smoothed_output_model = torch.nn.parallel.DataParallel(smoothed_output_model)
        smoothed_output_model_noddp = smoothed_output_model.module
        checkpoint_smooth = torch.load(simmim_path + checkpoint_path)
        msg = smoothed_output_model_noddp.load_state_dict(checkpoint_smooth['model'], strict=False)
        print(msg)
        print('************* smooth model loaded')
        for param in smoothed_output_model.parameters():
            param.requires_grad = False
        n_parameters = sum(p.numel() for p in smoothed_output_model.parameters() if p.requires_grad)
        print('parameters in the smoothed model, ', n_parameters)




    # Observe that all parameters are being optimized
    # delete , amsgrad=False
    # optimizer_ft = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr)
    # optimizer_ft = SGLD(model.parameters(), lr=args.lr, addnoise=True)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.lr_step, gamma=args.lr_gamma)

    # use mixed precision
    scaler = GradScaler()

    if args.ckpt is not None:
        ckpt_path = output_path + args.ckpt
        checkpoint = torch.load(ckpt_path)
        args.start_iter = int((args.ckpt).split('.')[-1])
        print('loading checkpoint', ckpt_path)
        print(args.start_iter)
        model.load_state_dict(checkpoint['model'])
        optimizer_ft.load_state_dict(checkpoint['optimizer'])
        exp_lr_scheduler.load_state_dict(checkpoint['scheduler'])
    # train model
    model = train_model(args, model, smoothed_output_model, criterion, optimizer_ft,exp_lr_scheduler, trainloader, testloader)


    torch.save(model, output_path+"gabor.resnet.final.%s"%args.iter)
