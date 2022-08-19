import numpy as np
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
from transformers import (
    TrainingArguments, Trainer,
    MT5Tokenizer
)
from datasets import load_dataset, Dataset,DatasetDict, load_metric
from model import (
    T5_finetuning
)
from transformers import (
    Seq2SeqTrainingArguments, Seq2SeqTrainer,DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM, AutoTokenizer, MT5Tokenizer, MT5ForConditionalGeneration
)

LOGGER = logging.getLogger()
def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    LOGGER.addHandler(console)
    
args = argparse.Namespace()
args.model_dir = '/Share/home/qiyifan/filebase/source/mt5-base'
args.data_path = '../data'
args.output_dir = '../tmp/mt5-base-finetuned'
args.max_length = 512
args.batch_size = 8
args.learning_rate = 2e-5
args.weight_decay = 0.0
args.use_cuda = True
args.parallel = True
args.epoch = 10
args.save_checkpoint_all = True

model = T5_finetuning(args)
tokenizer = MT5Tokenizer.from_pretrained('/Share/home/qiyifan/filebase/source/mt5-base')

from dataset import DaDataset
dataset = DaDataset(args, type = 'train.json')

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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    init_logging()

    if args.parallel:
        model.model = torch.nn.DataParallel(model.model)
        LOGGER.info("using nn.DataParallel")
    
    scaler = None

    start = time.time()
    step_global = 0
    for epoch in range(1,args.epoch+1):
        # embed dense representations for query and dictionary for train
        # Important! This is iterative process because dense represenation changes as model is trained.
        LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))
        data_loader = DataLoader(dataset, batch_size = args.batch_size, num_workers =16)
        # train
        loss, step_global = train(args, data_loader, model, scaler, step_global = step_global)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(loss,epoch))
        
        if args.save_checkpoint_all:
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

main(args)