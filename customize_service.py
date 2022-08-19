# -*- coding: utf-8 -*-
import logging
import os
import re
import torch
import json
from model_service.pytorch_model_service import PTServingBaseService
from transformers import (
    MT5ForConditionalGeneration, MT5Tokenizer
)

from flashtext import KeywordProcessor

logger = logging.getLogger(__name__)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class DaService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        super(PTServingBaseService, self).__init__(model_name, model_path)
        dir_path = os.path.dirname(os.path.realpath(model_path))
        bert_path = os.path.join(dir_path, 'checkpoint_63')
        self.model = MT5ForConditionalGeneration.from_pretrained(bert_path)
        self.model.to(DEVICE)
        self.tokenizer = MT5Tokenizer.from_pretrained(bert_path)
        self._pat_compile()
        
        ## label and key path
        label_path = os.path.join(dir_path, 'labels.txt')
        labelkey_path = os.path.join(dir_path, 'labels_hole.txt')
        
        ## keyword processor and labels
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        
        with open(label_path, 'r', encoding='utf-8') as f:
            self.labels = dict.fromkeys([i.strip('\n') for i in f.readlines()])
        with open(labelkey_path, 'r', encoding='utf-8') as f:
            labelkeys = set([line.strip('\n') for line in f.readlines()])
        for label in labelkeys:
            label = label.split('||')
            if len(label) == 1:
                self.keyword_processor.add_keyword(label[0])
            else:
                self.keyword_processor.add_keyword(label[0])
                for i in label[1:]:
                    self.keyword_processor.add_keyword(i, label[0])
        
    def _get_token(self, content, pad_size):
        tokenized_inputs = self.tokenizer.encode_plus(content, max_length=pad_size, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        return input_ids, attention_mask

    def _get_diagnosis(self, pred):
        pred_index = [i for i in range(len(pred)) if pred[i] == 1]
        pred_diagnosis = [self.id2label[index] for index in pred_index]
        return pred_diagnosis

    def _preprocess(self, data):
        # https两种请求形式
        # 1. form-data文件格式的请求对应：data = {"请求key值":{"文件名":<文件io>}}
        # 2. json格式对应：data = json.loads("接口传入的json体")
        data_dict = data.get('json_line')
        for v in data_dict.values():
            infer_dict = json.loads(v.read())
            return infer_dict

    def _inference(self, data):
        self.model.eval()
        emr_id = data.get('emr_id')
        text = self._preprocess_ehr(data)
        input_ids, attention_mask = self._get_token(text, 512)
        output = self.model.generate(input_ids = input_ids.to(DEVICE), attention_mask = attention_mask.to(DEVICE))
        infer = self.tokenizer.decode(output[0])
        ## infer :string
        result = {emr_id: infer}
        return result

    def _postprocess(self, data):
        infer_output = None
        ## k is id, v is str
        for k, v in data.items():
            v = self.pat3.sub("", v) 
            target = set([a.strip() for a in v.split(self.sep_token)])
            pred_diagnosis = [i for i in self.labels if i in target]
            infer_output = {k: pred_diagnosis}
        return infer_output

    ## add new function 
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
        
        pad_token = '<pad>'
        start_token = '</s>'
        
        self.sep_token = ','
        self.pat3 = re.compile(f'{pad_token}|{start_token}')