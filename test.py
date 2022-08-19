import numpy as np
import re
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

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

def get_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',type = str)
    parser.add_argument('--data_path',type = str)
    parser.add_argument('--output_dir',type = str)
    parser.add_argument('--max_length',type = int)
    parser.add_argument('--batch_size',type = int)
    parser.add_argument('--learning_rate',type = float, default = 2e-5)
    parser.add_argument('--weight_decay',type = float, default = 0.0)
    parser.add_argument('--use_cuda',action = 'store_true')
    parser.add_argument('--parallel',action = 'store_true')
    args = parser.parse_args()
    return args

def inference (args):
    model = T5_finetuning(args)
    tokenizer = model.tokenizer
    model.eval()
    if args.use_cuda:
        model.cuda()
    if args.parallel:
        model.model = torch.nn.DataParallel(model.model)
        
    testset = DaDataset(args, type = 'train_all.json')
    dataloader = DataLoader(testset, batch_size= args.batch_size, num_workers=4)
    
    outputs = []
    targets = []
    for i , batch in tqdm(enumerate(dataloader),total = len(dataloader)):
        if args.use_cuda:
            for k,v in batch.items():
                batch[k] = v.cuda()
        outs = model.model.generate(input_ids= batch["input_ids"], attention_mask = batch['attention_mask'])
        decode_outs = [tokenizer.decode(ids.detach().cpu().numpy()) for ids in outs]
        batch["labels"][batch["labels"][:, :] == -100] = tokenizer.pad_token_id
        target = [tokenizer.decode(ids) for ids in batch["labels"]]
        outputs.extend(decode_outs)
        targets.extend(target)
        
    return outputs, targets

def postprocessing(outputs, targets, mlb):
    pad_token = '<pad>'
    start_token = '</s>'
    sep_token = ','
    pattern = re.compile(f'{pad_token}|{start_token}')
    outputs = [pattern.sub("", i ) for i in outputs]
    targets = [pattern.sub("", i ) for i in targets]
    outputs_ = [set([a.strip() for a in i.split(sep_token)]) for i in outputs]
    targets_ = [set([a.strip() for a in i.split(sep_token)]) for i in targets]
    mlb.fit(targets_)
    outputs_id = mlb.transform(outputs_)
    targets_id = mlb.transform(targets_)  

    report = classification_report(targets_id, outputs_id, output_dict = True, target_names = mlb.classes_)
    report = pd.DataFrame().from_dict(report)
    report.T.to_excel(f'{args.output_dir}/report.xlsx')  
    
if __name__ == '__main__':
    args = get_args()
    outputs , targets = inference(args)
    mlb = MultiLabelBinarizer()
    # with open('../data/labels.txt','r') as f:
    #     labels = list(set(a for i in targets for a in i))
    # print(labels)
    # mlb.fit([labels])
    postprocessing(outputs, targets, mlb)
