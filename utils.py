import os 
import time
import random
import torch 
import math

def print_cz(str, f=None):
    if f is not None:
        print(str, file=f)
        if random.randint(0, 20) < 3:
            f.flush()
    print(str)

def time_mark():
    time_now = int(time.time())
    time_local = time.localtime(time_now)

    dt = time.strftime('%Y%m%d-%H%M%S', time_local)
    return(dt)

def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))

def model_snapshot(model, new_file, old_file=None, save_dir='./', verbose=True, log_file=None):
    """
    :param model: network model to be saved
    :param new_file: new pth name
    :param old_file: old pth name
    :param verbose: more info or not
    :return: None
    """
    from collections import OrderedDict
    import torch

    if os.path.exists(save_dir) is False:
        os.makedirs(expand_user(save_dir))
        print_cz(str='Make new dir:'+save_dir, f=log_file)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    for file in os.listdir(save_dir):
        if old_file in file:
            if verbose:
                print_cz(str="Removing old model  {}".format(expand_user(save_dir + file)), f=log_file)
            os.remove(save_dir + file) # 先remove旧的pth，再存储新的
    if verbose:
        print_cz(str="Saving new model to {}".format(expand_user(save_dir + new_file)), f=log_file)
    # torch.save({'cfg': cfg, 'state_dict': model.state_dict()}, expand_user(save_dir + new_file))
    torch.save(model, expand_user(save_dir + new_file))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0# value = current value
        self.avg = 0
        self.sum = 0# weighted sum
        self.count = 0# total sample num

    def update(self, value, n=1):# n是加权数
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
 
def dwt_init(x):
    """wavelet packet transform, as illustrated in Eq.2"""
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    """reverse wavelet packet transform"""
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h