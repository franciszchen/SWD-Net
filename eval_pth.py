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

import pytorch_ssim
import config
import data_loader_lmdb
from model import spatial_stage, wavelet_stage, swdnet
from utils import print_cz, AverageMeter, dwt_init


def test(testing_data_loader, model, criterion, epoch, logfile=None):

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    test_psnrs = AverageMeter()
    test_ssims = AverageMeter()
    end = time.time()
    test_start_time = time.time()
    with torch.no_grad():
        for batch in testing_data_loader:
            data, target = Variable(batch[0]).cuda(), Variable(batch[1], requires_grad=False).cuda()
            # data, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
            data_time.update(time.time() - end)

            output, _, _ = model(data)

            ssim_value = pytorch_ssim.ssim(output, target)
            test_ssims.update(ssim_value, target.shape[0])

            mse = torch.sum(torch.sum(((output[:, :, 8:-8, 8:-8]-target[:, :, 8:-8, 8:-8])*(output[:, :, 8:-8, 8:-8]-target[:, :, 8:-8, 8:-8])).view(-1, (target.shape[2]-16), (target.shape[3]-16)), dim=-1, keepdim=False), dim=-1, keepdim=False)/float((target.shape[2]-16)*(target.shape[3]-16))
            psnrs = 10*torch.log10(1.0/mse)
            test_psnrs.update((psnrs.mean()).data.item(), target.shape[0]*target.shape[1])

            del batch, data, output, target, mse, ssim_value, psnrs

            batch_time.update(time.time() - end)
            end = time.time()
        test_end_time = time.time()
        print_cz('  Test PSNR: {:.3f} dB \tSSIM: {:.3f}\t Time: {:.1f}\t BatchT: {:.3f}\t DataT: {:.3f}\t D/B: {:.1f}%'\
                .format(
                    test_psnrs.avg, 
                    test_ssims.avg, 
                    (test_end_time-test_start_time), 
                    batch_time.avg, 
                    data_time.avg, 
                    100.0*(data_time.avg/batch_time.avg)), 
                f=logfile)



def main():
    
    global opt, model
    opt = config.get_args()
    print(opt)
    
    log_file = None
    
    starting_time = time.time()
    print_cz(str=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), f=log_file) 

    cudnn.benchmark = True
    print_cz("===> Building model", f=log_file)
    spatial_sr = spatial_stage.Spatial_Stage()
    wavelet_sr = wavelet_stage.Wavelet_Stage()
    model = swdnet.SWDNet(spatial_sr, wavelet_sr)
    if opt.data_degradation in ['bicubic', 'Bicubic']:
        pth_file = './weights/swdnet-bicubic-dict.pth'
    elif opt.data_degradation in ['nearest', 'Nearest']:
        pth_file = './weights/swdnet-nearest-dict.pth'
    model.load_state_dict(torch.load(pth_file))

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
    testing_data_loader = data_loader_lmdb.get_loader(
        os.path.join(config.dataset_dir, opt.data_degradation, 'test_lmdb'), 
        batch_size=opt.batch_size, 
        stage='test', 
        num_workers=opt.num_workers)
    
    print_cz("===> Testing", f=log_file)
    test(testing_data_loader, model, criterion, epoch=0, logfile=None)

    print_cz(str(time.time()-starting_time), f=log_file)

if __name__ == "__main__":
    main()
