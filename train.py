import numpy as np
import json
import time
import argparse
import torch
import pandas as pd
import os
import logging
import sklearn.metrics as metrics
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import DaDataset
from datasets import load_dataset, Dataset,DatasetDict, load_metric
from model import (
    T5_finetuning
)
from transformers import (
    MT5Tokenizer, MT5ForConditionalGeneration
)

LOGGER = logging.getLogger()

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',type = str)
    parser.add_argument('--data_path',type = str)
    parser.add_argument('--output_dir',type = str)
    parser.add_argument('--max_length',type = int)
    parser.add_argument('--batch_size',type = int)
    parser.add_argument('--seed',type = int)
    parser.add_argument('--epoch',type = int)
    parser.add_argument('--learning_rate',type = float)
    parser.add_argument('--weight_decay',type = float)
    parser.add_argument('--use_cuda',action = 'store_true')
    parser.add_argument('--parallel',action = 'store_true')
    parser.add_argument('--save_checkpoint_all',action = 'store_true')
    args = parser.parse_args()
    return args

def train(args, data_loader, model, scaler,step_global = 0):
    LOGGER.info("train!")
    train_loss = 0
    train_steps = 0
    for i, batch in tqdm(enumerate(data_loader),total = len(data_loader)):
        model.optimizer.zero_grad()
        if args.use_cuda:
            model.cuda()
            model.train()
            batch_cuda ={}
            for k,v in batch.items():
                batch_cuda[k] = v.cuda()
        else:
            model.train()
            batch_cuda = batch
            
        loss = model(input_ids = batch_cuda['input_ids'], labels = batch_cuda['labels'])[0]
        loss.sum().backward()
        model.optimizer.step()        
        train_loss = train_loss + loss.sum().item()
        train_steps += 1
        step_global += 1
        if (i+1) % 10 == 0:
            LOGGER.info ("batches: {} loss: {:.3f}".format(i+1,train_loss / (train_steps+1e-9)))
            LOGGER.info ("batches: {} loss: {:.3f}".format(i+1, loss.sum().item()))
            
    train_loss /= (train_steps + 1e-9)
    return train_loss, step_global



def main(args):
    model = T5_finetuning(args)
    dataset = DaDataset(args, type = 'train_all.json')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
            
    init_logging()
    init_seed(args.seed)
    if args.parallel:
        model.model = torch.nn.DataParallel(model.model)
        LOGGER.info("using nn.DataParallel")
    
    scaler = None

    start = time.time()
    step_global = 0
    for epoch in range(1,args.epoch+1):
        # embed setence representations for query and target for train
        # Important! This is iterative process because dense represenation changes as model is trained.
        LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))
        data_loader = DataLoader(dataset, batch_size = args.batch_size, num_workers =16)
        loss, step_global = train(args, data_loader, model, scaler, step_global = step_global)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(loss,epoch))
        
        if args.save_checkpoint_all and epoch%2==1:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model.save_model(checkpoint_dir)

        if epoch == args.epoch:
            model.save_model(args.output_dir)
            
    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))

if __name__ == "__main__":
    args = get_args()
    main(args)