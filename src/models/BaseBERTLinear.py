import torch
from transformers import AutoModel


class BertForMultipleChoice(torch.nn.Module):
    def __init__(self, bert_name):
        super(BertForMultipleChoice, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.linear = torch.nn.Linear(self.bert.pooler.dense.out_features, 1)
        self.dropout = torch.nn.Dropout(p=0.10)
        self.num_choices = 4
        

    def forward(self, input_ids, attention_mask, token_type_ids):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        bert_output = self.bert(input_ids=flat_input_ids,
                                attention_mask=flat_attention_mask,
                                token_type_ids=flat_token_type_ids)
        pooler_output = self.dropout(bert_output.pooler_output)
        linear_output = self.linear(pooler_output)
        output_reshaped = linear_output.view(-1, self.num_choices)
        return output_reshaped
