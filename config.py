import torch
import argparse

save_dir = './save/'
dataset_dir = './HistoSR/'

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description="SWD-Net")
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--batch_size", type=int, default=24, help="Training batch size") # 8
    parser.add_argument("--epochs", type=int, default=600, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
    parser.add_argument("--step", type=int, default=150, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--init", type=str, default='No_init')

    parser.add_argument("--SWratio_tmp", type=float, default=0.1) # lambda1 in Eq.4
    parser.add_argument("--SWratio", type=float, default=0.1) # lambda2

    parser.add_argument("--interpolate", type=str, default='nearest')
    parser.add_argument("--corner", type=bool, default=None)    
    parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of threads for data loader to use, Default: 1")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
    parser.add_argument("--weight_decay", "--wd", default=1e-5, type=float, help="Weight decay, Default: 1e-4")
    parser.add_argument("--theme", type=str, default="")
    parser.add_argument("--log_path", type=str, default=save_dir)
    parser.add_argument("--job_type", type=str, default='S')
    # dataset path
    parser.add_argument('--data_degradation', default='bicubic', type=str, help="Choose HistoSR degradation, i.e., bicubic or nearest")

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    import json

    args = get_args()
    with open('./args.json', 'w') as f:
        f.write(json.dumps(args.__dict__, indent=4))

