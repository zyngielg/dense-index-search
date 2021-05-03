import torch
from transformers import AutoModel


class BaseBERTLinear(torch.nn.Module):
    def __init__(self, load_weights=False):
        super(BaseBERTLinear, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-cased')
        self.linear = torch.nn.Linear(self.bert.pooler.dense.out_features, 1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
        linear_output = self.linear(bert_output.pooler_output)
        return linear_output
