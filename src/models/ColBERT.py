import string
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from collections import OrderedDict

class ColBERT(BertPreTrainedModel):
    DEVICE = 'cpu'
    # note: ColBERT was using dim=128, but the checkpoint from huggingface requires 32

    def __init__(self, config, query_maxlen, doc_maxlen, device='cpu', mask_punctuation=True, dim=128, similarity_metric='cosine'):

        super(ColBERT, self).__init__(config)
        self.DEVICE = device

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

        self.bert = BertModel.from_pretrained(config._name_or_path).to(self.DEVICE)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False).to(self.DEVICE)

        self.init_weights()

    def forward(self, Q, D):
        return self.score(self.query(*Q), self.doc(*D))

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(
            self.DEVICE), attention_mask.to(self.DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=False):
        input_ids, attention_mask = input_ids.to(
            self.DEVICE), attention_mask.to(self.DEVICE)
        D = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids),
                            device=self.DEVICE).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def score(self, Q, D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d]
                for d in input_ids.cpu().tolist()]
        return mask

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
            print("[WARNING] Loading checkpoint with strict=False")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
