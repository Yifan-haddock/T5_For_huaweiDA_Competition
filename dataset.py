import numpy as np
import random
import pandas as pd
import json
import joblib
from torch.utils.data import Dataset
import re
import os 
from tqdm import tqdm
from transformers import (
    MT5Tokenizer, 
    MT5ForConditionalGeneration, 
)
from flashtext import KeywordProcessor

class DaDataset(Dataset):
    
    def __init__(self, args, type = 'train.json'):
        dataset_path = os.path.join(args.data_path,type)
        label_path = os.path.join(args.data_path,'labels.txt')
        with open(dataset_path,'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        with open(label_path,'r') as f:
            self.labels = set([line.strip('\n') for line in f.readlines()])
        self.max_len = args.max_length
        self.tokenizer = MT5Tokenizer.from_pretrained(args.model_dir)
        self.inputs = []
        self.targets = []
        self.prefix = 'diagnosis: '
        self.target_max_len = 64
        self._pat_compile()
        
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        for label in self.labels:
            label = label.split('||')
            if len(label) == 1:
                self.keyword_processor.add_keyword(label[0])
            else:
                self.keyword_processor.add_keyword(label[0])
                for i in label[1:]:
                    self.keyword_processor.add_keyword(i, label[0])
        
        self._whole_text_build()
  
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze
        target_ids[target_ids[:] == 0 ] = -100

        return {'input_ids':source_ids, "attention_mask":src_mask, "labels":target_ids,"labels_mask":target_mask}
    
    def _build(self):
        for idx in tqdm(range(len(self.lines))):
            data = json.loads(self.lines[idx])
            input_, target = data['history_of_present_illness'].strip(), data['diagnosis']
            input_ = self.prefix + input_
            target = ' , '.join(target).strip()
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len, padding='max_length', return_tensors="pt", truncation= True
            )
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length= self.target_max_len, padding='max_length', return_tensors="pt", truncation= True
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
            
    def _whole_text_build(self):
        pat_clean = self.pat_clean
        for idx in tqdm(range(len(self.lines))):
            data = json.loads(self.lines[idx])
            text = self._preprocess_ehr(data)
            input_ = pat_clean.sub("",text)
            target = data['diagnosis']
            input_ = self.prefix + input_
            target = ' , '.join(target).strip()
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len, padding='max_length', return_tensors="pt", truncation= True
            )
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length= self.target_max_len, padding='max_length', return_tensors="pt", truncation= True
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
            
    def _preprocess_ehr(self, data):
        chief_complaint = data['chief_complaint'].strip() if data['chief_complaint']!= None else ''
        history = data['history_of_present_illness'].strip() if data['history_of_present_illness']!= None else ''
        past_history = data['past_history'].strip() if data['past_history']!= None else ''
        physical_examination = data['physical_examination'].strip() if data['physical_examination']!= None else ''
        supplementary_examination = data['supplementary_examination'].strip() if data['supplementary_examination']!= None else ''
        age_ = data['age'].strip() if data['age']!= None else ''
        age = self._age_convert(age_)
        if age != '':
            age = '患者是'+age+ "。"
        text = chief_complaint + history +  past_history+ physical_examination+ supplementary_examination
        exist_diagnosis = list(dict.fromkeys(self.keyword_processor.extract_keywords(text)))
        if len(exist_diagnosis) == 0:
            return "可能的诊断：无。" + age + chief_complaint + history + '检查提示：' + supplementary_examination + physical_examination + '过去有：'+ past_history
        return "可能的诊断：" + ', '.join(exist_diagnosis) + '。' + age +chief_complaint + history + '检查提示：' + supplementary_examination + physical_examination + '过去有：'+ past_history
    
    def _age_convert(self, age_):
        if self.pat2.search(age_):
            age = '幼儿'
        else:
            if self.pat.search(age_):
                time = int(self.pat.search(age_).group())
                if time<=12:
                    age = '幼儿'
                elif 12<time<=30:
                    age = ''
                elif 30<time<=60:
                    age = ''
                else:
                    age = '老年'
            else:
                age = ''
        return age
        
    def _pat_compile(self,):
        self.pat = re.compile(r'\d+')
        self.pat2 = re.compile(r'[时天月]')
        self.pat_clean = re.compile(r'\n')