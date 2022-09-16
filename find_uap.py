import time
import torch
import torchvision
import numpy as np
import torch.optim as optim
import os
import argparse
from utils import * 

def get_aug():
    parser = argparse.ArgumentParser(description='Feature-Gathering Universal Adversarial Perturbation')
    parser.add_argument('--remark', default='', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=2, type=int)
    parser.add_argument('--model_name', default='vgg16', type=str,
        help='Choose from "alexnet, googlenet, vgg16, vgg19, resnet50, resnet152, deit_tiny, deit_small, deit_base".')
    parser.add_argument('--train_data_dir', default='path_of_train_data', type=str)
    parser.add_argument('--val_data_dir', default='path_of_validation_data', type=str)
    parser.add_argument('--result_dir', default='path_of_result_dir', type=str)
    parser.add_argument('--xi', default=0.0392, type=float)
    parser.add_argument('--p', default=np.inf, type=float)
    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--top_k', default=[1,3,5], nargs='+', type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--target', default=-1, type=int)
    parser.add_argument('--target_param', default=0.1, type=float)
    parser.add_argument('--val_freq', default=1, type=int)


    args = parser.parse_args()
    return args

def create_logger(args):
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    exp_time = time.strftime('%m%d_%H%M')
    log_dir = os.path.join(args.result_dir, exp_time + '_' + args.model_name + '_' + str(args.target))
    if args.remark:
        log_dir += '_' + args.remark
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = my_logger(args, os.path.join(log_dir, 'log.txt'))
    logger.info(args)
    return logger, log_dir

def main():
    start_time = time.time()
    args = get_aug()

    logger, log_dir = create_logger(args)

    torch.cuda.set_device(int(args.gpu))
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    net = get_model(args.model_name)
    net = net.cuda()

    train_loader, val_loader = get_data(args.train_data_dir, args.val_data_dir, args.batch_size)
    attacker = FG_UAP(args.xi, args.p, net, logger, args.target, args.target_param)
    uap, fr = attacker.attack(train_loader, val_loader, args.max_epoch, args.lr, args.top_k, args.val_freq)
    logger.info('Best FR: {:.2f}. Total time: {:.2f}\n'.format(fr, time.time()-start_time))
    uap = uap.data.cpu()
    torch.save(uap, os.path.join(log_dir, args.model_name + '_{:.2f}.pth'.format(fr)))


if __name__ == '__main__':
    main()