import string
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from collections import OrderedDict
from models.colbert_parameters import DEVICE

class ColBERT(BertPreTrainedModel):
    # DEVICE = 'cuda:3'
    # note: ColBERT was using dim=128, but the checkpoint from huggingface requires 32

    def __init__(self, config, query_maxlen, doc_maxlen, device='cpu', mask_punctuation=True, dim=128, similarity_metric='cosine'):

        super(ColBERT, self).__init__(config)
        self.DEVICE = DEVICE

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim
        self.mask_punctuation = mask_punctuation
        self.skiplist = {}
        
        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained(config._name_or_path)
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.bert = BertModel.from_pretrained(config._name_or_path)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

        # self.init_weights()

    def forward(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(
            self.DEVICE), attention_mask.to(self.DEVICE)

        bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]
        output_reduced = self.linear(bert_output)

        return output_reduced

    @staticmethod
    def load_checkpoint(path, model):
        print(f"******** Loading checkpoint from {path} ********")

        checkpoint = torch.load(path, map_location='cpu')

        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if k[:7] == 'module.':
                name = k[7:]
            new_state_dict[name] = v

        checkpoint['model_state_dict'] = new_state_dict

        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:            
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
