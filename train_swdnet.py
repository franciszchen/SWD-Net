import argparse, os
import torch
import random
import time
import json
import numpy as np
import math
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import config
import data_loader_lmdb
import pytorch_ssim
from model import spatial_stage, wavelet_stage, swdnet
from utils import print_cz, time_mark, expand_user, model_snapshot, AverageMeter, dwt_init

def prepare():
    """
    config, make dirs
    """
    time_tag = time_mark()
    log_dir = opt.log_path + time_tag + '_' + opt.theme + '_' + opt.optim + '_lr' + str(opt.lr) + '_wd'+str(opt.weight_decay) +\
        '_bs'+str(opt.batch_size)+'_epochs'+str(opt.epochs) + '_step'+str(opt.step) +'_gamma'+str(opt.gamma)+'_nw'+str(opt.num_workers)
    if os.path.exists(log_dir) is False:# make dir if not exist
        os.makedirs(expand_user(log_dir))
        print('make dir: ' + str(log_dir))
    return log_dir

def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by gamma every step epochs"""
    lr = opt.lr * (opt.gamma ** (epoch // opt.step))
    return lr

def train(
    training_data_loader, 
    testing_data_loader, 
    swratio, 
    swratio_tmp, 
    optimizer, 
    optimizer_spatial,
    model, 
    criterion,
    save_dir=None, 
    logfile=None, 
    pth_prefix=''):
    print_cz("===> Training Spatial Stage", f=logfile)
    for epoch in range(opt.epochs):
        lr = adjust_learning_rate(epoch)
        for param_group in optimizer_spatial.param_groups:
            param_group["lr"] = lr
        print_cz("Epoch = {}, lr = {}".format(epoch, optimizer_spatial.param_groups[0]["lr"]), f=logfile)
        train_a_epoch_spatial(
            training_data_loader=training_data_loader,  
            optimizer=optimizer_spatial, 
            model=model, 
            criterion=criterion, 
            epoch=epoch, 
            logfile=logfile)
        test_spatial(
            testing_data_loader=testing_data_loader, 
            model=model, 
            criterion=criterion, 
            epoch=epoch, 
            logfile=logfile)
            
    print_cz("===> Training SWD-Net", f=logfile)
    best_psnr = 0
    test_spatial(
            testing_data_loader=testing_data_loader, 
            model=model, 
            criterion=criterion, 
            epoch=epoch, 
            logfile=logfile)
    for epoch in range(opt.epochs):
        lr = adjust_learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print_cz("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]), f=logfile)
        train_loss, train_psnr = train_a_epoch(
            training_data_loader, 
            swratio, 
            swratio_tmp, 
            optimizer, 
            model, 
            criterion, 
            epoch, 
            logfile)
        test_loss, test_psnr, test_ssim = test(
            testing_data_loader, 
            model, 
            criterion,  
            epoch, 
            logfile)
        if test_psnr > best_psnr:
            best_psnr = test_psnr
            if save_dir is not None: # save flag
                model_snapshot(model, new_file=(
                    pth_prefix+'model-best-{}-TestL{:.1f}-TestPSNR-{:.3f}dB-TestSSIM{:.3f}-{}.pth'.format(
                        epoch,
                        test_loss, 
                        best_psnr, 
                        test_ssim, 
                        time_mark())),
                    old_file=pth_prefix + 'model-best-', save_dir=save_dir+'/', verbose=True)
                print_cz('*better model saved successfully*', f=logfile)


def train_a_epoch_spatial(
    training_data_loader,  
    optimizer, 
    model, 
    criterion, 
    epoch, 
    logfile):
    
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_losses = AverageMeter()
    train_psnrs = AverageMeter()

    end = time.time()
    train_start_time = time.time()
    for iteration, batch in enumerate(training_data_loader):
        # (96) and (192)
        data, target = Variable(batch[0]).cuda(), Variable(batch[1], requires_grad=False).cuda()
        data_time.update(time.time() - end)
        _, _, output = model(data) # train spatial stage
        # compute loss
        loss = criterion(output, target)
        
        mse = torch.sum(torch.sum(((output[:, :, 8:-8, 8:-8]-target[:, :, 8:-8, 8:-8])*(output[:, :, 8:-8, 8:-8]-target[:, :, 8:-8, 8:-8])).view(-1, (target.shape[2]-16), (target.shape[3]-16)), dim=-1, keepdim=False), dim=-1, keepdim=False)/float((target.shape[2]-16)*(target.shape[3]-16))
        psnrs = 10*torch.log10(1.0/mse)
        train_psnrs.update((psnrs.mean()).data.item(), target.shape[0]*target.shape[1])
        # update model
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),opt.clip) # new
        optimizer.step()

        train_losses.update(loss.data.item(), target.shape[0])
        if iteration % 100 == 1:
            print_cz('  Batch {:d}, loss {:.1f}, PSNR_present: {:.3f} dB, PSNR_avg: {:.3f} dB'\
                .format(iteration, loss.data.item(), psnrs.mean(), train_psnrs.avg ), f=logfile)

        del batch, data, output, target, loss, mse, psnrs

        batch_time.update(time.time() - end)
        end = time.time()
    train_end_time = time.time()
    print_cz('  Train Loss: {:.3f}\t PSNR: {:.3f} dB\t Time: {:.1f}\t BatchT: {:.3f}\t DataT: {:.3f}\t D/B: {:.1f}%'\
        .format(train_losses.avg, train_psnrs.avg, (train_end_time-train_start_time), batch_time.avg, data_time.avg, 100.0*(data_time.avg/batch_time.avg)), f=logfile)


def train_a_epoch(
    training_data_loader, 
    swratio, 
    swratio_tmp, 
    optimizer, 
    model, 
    criterion, 
    epoch, 
    logfile):
    
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_losses = AverageMeter()
    train_losses_spatial_tmp = AverageMeter()
    train_losses_spatial = AverageMeter()
    train_losses_wavelet = AverageMeter()
    train_psnrs = AverageMeter()

    end = time.time()
    train_start_time = time.time()
    for iteration, batch in enumerate(training_data_loader):
        # (96) and (192)
        data, target = Variable(batch[0]).cuda(), Variable(batch[1], requires_grad=False).cuda()
        data_time.update(time.time() - end)

        output, wave_output, spatial_output = model(data) 
        # compute loss
        loss_spatial_tmp = criterion(spatial_output, target)
        loss_spatial = criterion(output, target)
        loss_wavelet = criterion(wave_output, dwt_init(target))
        loss = swratio_tmp * loss_spatial_tmp + swratio * loss_spatial + loss_wavelet # Eq.4
        
        mse = torch.sum(torch.sum(((output[:, :, 8:-8, 8:-8]-target[:, :, 8:-8, 8:-8])*(output[:, :, 8:-8, 8:-8]-target[:, :, 8:-8, 8:-8])).view(-1, (target.shape[2]-16), (target.shape[3]-16)), dim=-1, keepdim=False), dim=-1, keepdim=False)/float((target.shape[2]-16)*(target.shape[3]-16))
        psnrs = 10*torch.log10(1.0/mse)
        train_psnrs.update((psnrs.mean()).data.item(), target.shape[0]*target.shape[1])
        # update model
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),opt.clip) # new
        optimizer.step()

        train_losses.update(loss.data.item(), target.shape[0])
        train_losses_spatial_tmp.update(loss_spatial_tmp.data.item(), target.shape[0])
        train_losses_spatial.update(loss_spatial.data.item(), target.shape[0])
        train_losses_wavelet.update(loss_wavelet.data.item(), target.shape[0])
        if iteration % 100 == 1:
            print_cz('  Batch {:d}, loss {:.1f}, wloss {:.1f}, sloss {:.1f}, sloss_tmp {:.1f}, PSNR_present: {:.3f} dB, PSNR_avg: {:.3f} dB'\
                .format(iteration, loss.data.item(), loss_wavelet.data.item(), loss_spatial.data.item(), loss_spatial_tmp.data.item(), psnrs.mean(), train_psnrs.avg ), f=logfile)

        del batch, data, output, target, loss, mse, psnrs, wave_output, spatial_output, loss_wavelet, loss_spatial, loss_spatial_tmp

        batch_time.update(time.time() - end)
        end = time.time()
    train_end_time = time.time()
    print_cz('  Train Loss: {:.3f}\t wLoss: {:.3f}\t sLoss: {:.3f}\t sLoss_tmp: {:.3f}\t PSNR: {:.3f} dB\t Time: {:.1f}\t BatchT: {:.3f}\t DataT: {:.3f}\t D/B: {:.1f}%'\
        .format(train_losses.avg, train_losses_wavelet.avg, train_losses_spatial.avg, train_losses_spatial_tmp.avg, train_psnrs.avg, (train_end_time-train_start_time), batch_time.avg, data_time.avg, 100.0*(data_time.avg/batch_time.avg)), f=logfile)
    return train_losses.avg, train_psnrs.avg


def test_spatial(
    testing_data_loader, 
    model, 
    criterion, 
    epoch, 
    logfile=None):

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    test_losses = AverageMeter()
    test_psnrs = AverageMeter()
    test_ssims = AverageMeter()
    end = time.time()
    test_start_time = time.time()
    with torch.no_grad():
        for batch in testing_data_loader:
            data, target = Variable(batch[0]).cuda(), Variable(batch[1], requires_grad=False).cuda()
            data_time.update(time.time() - end)
            _, _, output = model(data)
            # compute loss
            loss = criterion(output, target)
            test_losses.update(loss.data.item(), target.shape[0])

            ssim_value = pytorch_ssim.ssim(output, target)
            test_ssims.update(ssim_value, target.shape[0])

            mse = torch.sum(torch.sum(((output[:, :, 8:-8, 8:-8]-target[:, :, 8:-8, 8:-8])*(output[:, :, 8:-8, 8:-8]-target[:, :, 8:-8, 8:-8])).view(-1, (target.shape[2]-16), (target.shape[3]-16)), dim=-1, keepdim=False), dim=-1, keepdim=False)/float((target.shape[2]-16)*(target.shape[3]-16))
            psnrs = 10*torch.log10(1.0/mse)
            test_psnrs.update((psnrs.mean()).data.item(), target.shape[0]*target.shape[1])

            del batch, data, output, target, loss, mse, ssim_value, psnrs

            batch_time.update(time.time() - end)
            end = time.time()
        test_end_time = time.time()
        print_cz('  Test Loss: {:.3f}\t PSNR: {:.3f} dB\t SSIM: {:.3f}\t Time: {:.1f}\t BatchT: {:.3f}\t DataT: {:.3f}\t D/B: {:.1f}%'\
                .format(test_losses.avg, test_psnrs.avg, test_ssims.avg, (test_end_time-test_start_time), batch_time.avg, data_time.avg, 100.0*(data_time.avg/batch_time.avg)), f=logfile)

def test(
    testing_data_loader, 
    model, 
    criterion, 
    epoch, 
    logfile=None):

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    test_losses = AverageMeter()
    test_psnrs = AverageMeter()
    test_ssims = AverageMeter()
    end = time.time()
    test_start_time = time.time()
    with torch.no_grad():
        for batch in testing_data_loader:
            data, target = Variable(batch[0]).cuda(), Variable(batch[1], requires_grad=False).cuda()
            data_time.update(time.time() - end)

            output, wave_output, spatial_output = model(data)
            # compute loss
            loss = criterion(wave_output, dwt_init(target))
            test_losses.update(loss.data.item(), target.shape[0])

            ssim_value = pytorch_ssim.ssim(output, target)
            test_ssims.update(ssim_value, target.shape[0])

            mse = torch.sum(torch.sum(((output[:, :, 8:-8, 8:-8]-target[:, :, 8:-8, 8:-8])*(output[:, :, 8:-8, 8:-8]-target[:, :, 8:-8, 8:-8])).view(-1, (target.shape[2]-16), (target.shape[3]-16)), dim=-1, keepdim=False), dim=-1, keepdim=False)/float((target.shape[2]-16)*(target.shape[3]-16))
            psnrs = 10*torch.log10(1.0/mse)
            test_psnrs.update((psnrs.mean()).data.item(), target.shape[0]*target.shape[1])

            del batch, data, output, target, loss, mse, ssim_value, psnrs, wave_output, spatial_output

            batch_time.update(time.time() - end)
            end = time.time()
        test_end_time = time.time()
        print_cz('  Test Loss: {:.3f}\t PSNR: {:.3f} dB\t SSIM: {:.3f}\t Time: {:.1f}\t BatchT: {:.3f}\t DataT: {:.3f}\t D/B: {:.1f}%'\
                .format(test_losses.avg, test_psnrs.avg, test_ssims.avg, (test_end_time-test_start_time), batch_time.avg, data_time.avg, 100.0*(data_time.avg/batch_time.avg)), f=logfile)
    
    return test_losses.avg, test_psnrs.avg, test_ssims.avg



def main():
    global opt, model
    opt = config.get_args()
    print(opt)
    save_folder = prepare()
    log_file = open((save_folder + '/' + 'print_out_screen.txt'), 'w')

    with open(save_folder + '/args.json', 'w') as f:
        f.write(json.dumps(opt.__dict__, indent=4))

    starting_time = time.time()
    print_cz(str=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), f=log_file) 
    # opt.seed = random.randint(1, 10000)
    # opt.seed = 127
    cudnn.benchmark = True

    print_cz("===> Building model", f=log_file)
    # print_cz(opt.interpolate + ' \t ' + str(opt.corner), f=log_file)
    spatial_sr = spatial_stage.Spatial_Stage()
    wavelet_sr = wavelet_stage.Wavelet_Stage()
    model = swdnet.SWDNet(spatial_sr, wavelet_sr)

    print_cz("===> Setting GPU", f=log_file)
    if opt.job_type == 'S' or opt.job_type == 's':
        model.cuda()
    else:
        if opt.job_type == 'Q' or opt.job_type == 'q':
            gpu_device_ids=[0, 1, 2, 3]
        elif opt.job_type == 'E' or opt.job_type == 'e':
            gpu_device_ids=[0, 1, 2, 3, 4, 5, 6, 7]
        elif opt.job_type == 'D' or opt.job_type == 'd':
            gpu_device_ids=[0, 1]
        model = nn.DataParallel(model.cuda(), device_ids=gpu_device_ids).cuda()

    criterion = nn.L1Loss(size_average=False)
  
    print_cz("===> Loading datasets", f=log_file)
    training_data_loader = data_loader_lmdb.get_loader(
        os.path.join(config.dataset_dir, opt.data_degradation, 'train_lmdb'), 
        batch_size=opt.batch_size, 
        stage='train', 
        num_workers=opt.num_workers)
    testing_data_loader = data_loader_lmdb.get_loader(
        os.path.join(config.dataset_dir, opt.data_degradation, 'test_lmdb'), 
        batch_size=opt.batch_size, 
        stage='test', 
        num_workers=opt.num_workers)
    
    print_cz("===> Setting Optimizer", f=log_file)
    if opt.optim in ['SGD', 'sgd']:
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
        optimizer_spatial = optim.SGD(model.spatial.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optim in ['Adam', 'adam']:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        optimizer_spatial = optim.Adam(model.spatial_sr.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    print_cz("===> Training", f=log_file)
    train(
        training_data_loader=training_data_loader, 
        testing_data_loader=testing_data_loader, 
        swratio=opt.SWratio, 
        swratio_tmp=opt.SWratio_tmp, 
        optimizer=optimizer, 
        optimizer_spatial=optimizer_spatial,
        model=model, 
        criterion=criterion, 
        save_dir=save_folder, 
        logfile=log_file)

    print_cz(str(time.time()-starting_time), f=log_file)
    log_file.close()


if __name__ == "__main__":
    main()
