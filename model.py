import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn
import logging
from transformers import (
    MT5ForConditionalGeneration, 
    MT5Tokenizer,
)

LOGGER = logging.getLogger()

class T5_finetuning(nn.Module):
    def __init__(self, hparam):
        super(T5_finetuning, self).__init__()
        self.hparam = hparam
        self.model = MT5ForConditionalGeneration.from_pretrained(hparam.model_dir)
        self.tokenizer = MT5Tokenizer.from_pretrained(hparam.model_dir)
        self.learning_rate = hparam.learning_rate
        self.weight_decay = hparam.weight_decay
        self.max_length = hparam.max_length
        self.use_cuda = hparam.use_cuda
        self.parallel = hparam.parallel
        self.optimizer = optim.AdamW(
            [{'params':self.model.parameters()}],
            lr = self.learning_rate, weight_decay = self.weight_decay
        )
    
    def forward(self, input_ids=None ,attention_mask=None,labels=None,labels_mask=None):
        
        output = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    return_dict = True
                )
        loss, logit = output['loss'], output['logits']
        
        return loss, logit
    
    def save_model(self,path):
        if self.parallel:
            self.model.module.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
            
        self.tokenizer.save_pretrained(path)