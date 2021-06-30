import string
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from collections import OrderedDict
from models.colbert_parameters import DEVICE

class ColBERT(BertPreTrainedModel):
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

        self.init_weights()

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]
        output_reduced = self.linear(bert_output)
        return output_reduced

    @staticmethod    
    def load_checkpoint(path, model, base=True):
        print(f"******** Loading checkpoint from {path} ********")
        if base:
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
        else:
            checkpoint = torch.load(path)
            state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}

    
        try:
            model.load_state_dict(state_dict)
        except:            
            model.load_state_dict(state_dict, strict=False)
