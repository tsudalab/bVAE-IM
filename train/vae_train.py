import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import pickle as pickle

from model import *
import rdkit
from tqdm import tqdm
import os

def main_vae_train(train,
             vocab,
             save_dir,
             binary_size=300,
             batch_size=32,
             depthT=20,
             depthG=3,
             lr=1e-3,
             clip_norm=50.0,
             beta=0.001,
             epoch=20,
             anneal_rate=0.9,
             anneal_iter=40000, 
             init_temp=1,
             temp_anneal_rate=1e-4,
             min_temp=0.4,
             print_iter=50, 
             save_iter=5000):
    
    vocab = [x.strip("\r\n ") for x in open(vocab)] 
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, int(binary_size), int(depthT), int(depthG)).cuda()
    print(model)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)

    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, anneal_rate)

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = 0
    beta = beta
    meters = np.zeros(4)
    temp = init_temp
    
    for epoch in tqdm(range(epoch)):
        loader = MolTreeFolder(train, vocab, batch_size)#, num_workers=4)
        for batch in loader:
            total_step += 1
            temp = max(min_temp, temp*np.exp(-temp_anneal_rate*total_step))
            # try:
            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc = model(batch, beta, temp)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

            if total_step % print_iter == 0:
                meters /= print_iter
                print("[%d] Temp: %.3f Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" \
                    % (total_step, temp, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model)))
                meters *= 0

            if total_step % save_iter == 0:
                torch.save(model.state_dict(), save_dir + "/model.iter-" + str(total_step))

            if total_step % anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])
    
        scheduler.step()

    return model


if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--save_dir', required=True)

    parser.add_argument('--binary_size', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=50.0)
    parser.add_argument('--beta', type=float, default=0.001)

    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--anneal_rate', type=float, default=0.9)
    parser.add_argument('--anneal_iter', type=int, default=40000)
    parser.add_argument('--init_temp', type=float, default=1.0)
    parser.add_argument('--temp_anneal_rate', type=float, default=1e-4)
    parser.add_argument('--min_temp', type=float, default=0.4)
    parser.add_argument('--print_iter', type=int, default=50)
    parser.add_argument('--save_iter', type=int, default=5000)

    args = parser.parse_args()
    print(args)
    
    main_vae_train(args.train,
             args.vocab,
             args.save_dir,
             args.binary_size,
             args.batch_size,
             args.depthT,
             args.depthG,
             args.lr,
             args.clip_norm,
             args.beta,
             args.epoch, 
             args.anneal_rate,
             args.anneal_iter, 
             args.init_temp,
             args.temp_anneal_rate,
             args.min_temp,
             args.print_iter, 
             args.save_iter)
    
