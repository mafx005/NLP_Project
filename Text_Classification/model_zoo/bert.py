# -*- coding: utf-8 -*-
# @Time : 2021/8/24 17:41
# @Author : M
# @FileName: bert.py
# @Dec :

import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, label=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooler_output = outputs['pooler_output']

        out = self.fc(pooler_output)

        # if label is not None:
        #     loss_fn = nn.BCEWithLogitsLoss()
        #     loss = loss_fn(logits, label)
        #     return loss, logits

        return out
