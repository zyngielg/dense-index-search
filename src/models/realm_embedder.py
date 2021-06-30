from transformers import BertTokenizerFast, BertModel, BertPreTrainedModel
from utils.huggingface_mlmhead import BertOnlyMLMHead

import torch


class REALMEmbedder(BertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        # self.dense = torch.nn.Linear(config.hidden_size, 128)
        self.dense = torch.nn.Linear(config.hidden_size, 128)
        self.LayerNorm = torch.nn.LayerNorm(128)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        bert_output = bert_output[0][:, 0, :]
        cls = self.cls(bert_output)
        projected = self.dense(cls)
        return self.LayerNorm(projected)
