from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from importlib import import_module
from model import robust_models
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sys


class my_logger:
    def __init__(self, args, log_dir):
        self.name = log_dir
        with open(self.name, 'w') as F:
            print('\n'.join(
                ['%s:%s' % item for item in args.__dict__.items() if item[0][0] != '_']), file=F)
            print('\n', file=F)

    def info(self, content):
        with open(self.name, 'a') as F:
            print(content)
            print(content, file=F)


class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]

        return x

class FG_UAP(object):

    def __init__(self, xi, p, model, logger, target, target_param):
        self.xi = xi
        self.p = p
        self.v = torch.autograd.Variable(torch.zeros((1, 3, 224, 224)).cuda(), requires_grad=True)
        self.model = model
        self.logger = logger
        self.target = target
        self.target_param = target_param

    def proj_lp(self):
        # Project on the lp ball centered at 0 and of radius xi
        if self.p == np.inf:
            self.v.data = torch.clamp(self.v.data, -self.xi, self.xi)
        else:
            self.v.data = self.v.data * min(1, self.xi / (torch.norm(self.v, self.p)+0.00001))
    
    def attack(self, train_loader, val_loader, max_epoch, lr, top_k, val_freq):
        best_v = torch.zeros_like(self.v)
        fooling_rate, best_fr = 0.0, 0.0
        optimizer = optim.Adam([self.v], lr=lr, weight_decay=1e-5)
        global layer_out 
        self.model = set_hooks(self.model, self.logger)

        for epoch in range(max_epoch):
            pbar = tqdm(train_loader)
            for img, label in pbar:
                optimizer.zero_grad()
                img, label = img.cuda(), label.cuda()
                with torch.enable_grad():
                    # get the logit and feature-layer out of clean image
                    layer_out[0] = None
                    i_out = self.model(img)
                    i_layer_out = layer_out[0].clone()
                    # get the logit and feature-layer out of adversarial image
                    layer_out[0] = None
                    adv_out = self.model(img + self.v)
                    adv_layer_out = layer_out[0].clone()
                    loss = cal_cossim(adv_layer_out, i_layer_out)
                    if self.target != -1:  # targeted FG-UAP
                        target_logits = F.cross_entropy(
                            adv_out, torch.tensor(self.target).repeat(adv_out.shape[0]).cuda())
                        loss += self.target_param * target_logits
                    else:
                        target_logits = torch.tensor(-1)
                loss.backward()
                optimizer.step()
                # clamp uap
                self.proj_lp()
                pbar.set_description('Epoch {} loss ({:.3f}) target_logit({:.3f})'.format(
                    epoch, loss.item(), target_logits.item()))
            if (epoch + 1) % val_freq == 0:
                fooling_rate = self.evaluate(val_loader, top_k)
                if fooling_rate > best_fr:
                    best_fr = fooling_rate
                    best_v = self.v.clone()
                
        return best_v, best_fr


    def evaluate(self, val_loader, top_k):
        global layer_out
        mean = [0.485, 0.456, 0.406]
        mean_tensor = torch.tensor(mean).unsqueeze(
            0).unsqueeze(-1).unsqueeze(-1).cuda()
        diff_all, ori_acc_all, adv_acc_all = 0, 0, 0
        adv_distribution = 0
        layer_out[0] = None
        v_out = self.model(self.v + mean_tensor)
        uap_class = v_out.argmax(-1).squeeze()
        self.logger.info('uap class: {}'.format(uap_class))
        v_layer_out = layer_out[0].clone()
        pbar = tqdm(val_loader)
        for img, label in pbar:
            img, label = img.cuda(), label.cuda()
            with torch.no_grad():
                # get the logit and feature-layer out of clean image
                layer_out[0] = None
                i_out = self.model(img)
                i_layer_out = layer_out[0].clone()
                # get the logit and feature-layer out of adversarial image
                layer_out[0] = None
                adv_out = self.model(img + self.v)
                adv_layer_out = layer_out[0].clone()
                loss = cal_cossim(adv_layer_out, i_layer_out)
                if self.target != -1: # targeted FG-UAP
                    target_logits = F.cross_entropy(adv_out, torch.tensor(self.target).repeat(adv_out.shape[0]).cuda())
                else:
                    target_logits = torch.tensor(-1)
                # for calculating dominance
                ori_pred = i_out.argmax(dim=1)
                adv_pred = adv_out.argmax(dim=1)
                adv_distribution += (adv_out == adv_out.max(1)
                                    [0].reshape(-1, 1)).sum(0)
                diff_all += (ori_pred != adv_pred).sum()
                ori_acc_all += (ori_pred == label).sum()
                adv_acc_all += (adv_pred == label).sum()
            pbar.set_description('Testing')
        total_num = adv_distribution.sum()
        fooling_rate = diff_all / total_num * 100
        self.logger.info('Test: FR: {:.2f} loss: {:.3f} target_logit: {:.3f}'.format(
            fooling_rate, (loss / len(pbar)).item(), (target_logits / len(pbar)).item()))

        if len(top_k) > 0: # calculate the dominance ratio
            total_num = adv_distribution.sum()
            pert_top = adv_distribution.topk(max(top_k), -1, True, True)
            out_dic = {}
            out_dic['ori_acc'] = ori_acc_all / total_num
            out_dic['adv_acc'] = adv_acc_all / total_num
            for k in top_k:
                res = uap_class in pert_top[1][:k]
                out_dic['DP_in_top{}'.format(k)] = int(res)
                out_dic['pert_occupy(top{})%'.format(k)] = float(100.0*pert_top[0][:k].sum()/total_num)

            for key in out_dic:
                self.logger.info('{}:{:.2f}'.format(key, out_dic[key]))

        return fooling_rate



def set_hooks(model, logger):
    global layer_out
    layer_out = [0]
    layer_out[0] = None

    def get_norm_input(self, forward_input, forward_output):
        try:
            layer_out[0] = forward_input[0] if layer_out[0] is None else torch.cat(
                (layer_out[0], forward_input[0]), dim=0)
        except:
            pass

    layers_to_opt = get_layers_to_opt(
        model[1].__class__.__name__)
    logger.info('hook layers:')
    for name, layer in model[1].named_modules():
        if (name in layers_to_opt):
            logger.info(name)
            layer.register_forward_hook(get_norm_input)
    return model
    


def get_data(train_data_dir, val_data_dir, batch_size):
    train_dataset = datasets.ImageFolder(
        root=train_data_dir,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
        ]))
    val_dataset = datasets.ImageFolder(
        root=val_data_dir,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    return train_loader, val_loader


def get_model(model):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    network_file = import_module('torchvision.models')
    if model.startswith('deit'):
        network_file = import_module('model.DeiT')
        net = getattr(network_file, '{}_patch16_224'.format(model))(
            pretrained=True)
    else:
        net = getattr(network_file, model)(pretrained=True)

    for params in net.parameters():
        params.requires_grad = False
    net = torch.nn.Sequential(
        Normalize(mean, std),
        net
    )
    net.eval()
    return net


def cal_cossim(out1, out2): 
    n_out1 = len(out1)
    n_out2 = len(out2)
    if n_out1 == n_out2:
        return torch.cosine_similarity(out1, out2, dim=1).mean()
    elif n_out1 > n_out2:
        batch = int(n_out1 / n_out2)
        sim = torch.tensor(0).float().cuda()
        for i in range(n_out2):
            sim += torch.cosine_similarity(
                out1[i * batch:(i + 1) * batch], out2[i].unsqueeze(0), dim=1).mean()
        return torch.mean(sim)
    else:
        raise('Please reverse the input order from cal_cossim(a, b) to cal_cossim(b, a).')


def get_layers_to_opt(model):

    if model == 'VisionTransformer':
        layers_to_opt = ['head']

    elif model == 'PoolingTransformer':
        layers_to_opt = ['head']

    elif model == 'VGG' or 'AlexNet' in model:
        layers_to_opt = ['classifier.6']

    elif 'ResNet' in model or 'GoogLeNet' in model:
        layers_to_opt = ['fc']

    return layers_to_opt
