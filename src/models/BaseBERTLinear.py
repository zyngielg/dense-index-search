import torch
from transformers import AutoModel


class BaseBERTLinear(torch.nn.Module):
    def __init__(self, load_weights=False):
        super(BaseBERTLinear, self).__init__()
        # bert_name = "bert-base-cased"
        bert_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        self.bert = AutoModel.from_pretrained(bert_name)
        self.linear = torch.nn.Linear(self.bert.pooler.dense.out_features, 1)
        self.dropout = torch.nn.Dropout(p=0.15)
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # bert_output = self.bert(input_ids=input_ids,
        #                         attention_mask=attention_mask,
        #                         token_type_ids=token_type_ids)
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        
        bert_output = self.bert(input_ids=flat_input_ids,
                                attention_mask=flat_attention_mask,
                                token_type_ids=flat_token_type_ids)
        linear_output = self.linear(bert_output.pooler_output)
        output_reshaped = linear_output.view(-1, 4)
        return output_reshaped
