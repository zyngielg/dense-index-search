import torch
from transformers import AutoModel, AutoTokenizer

class CustomBERT(torch.nn.Module):
    def __init__(self):
          super(CustomBERT, self).__init__()
          self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
          self.bert = AutoModel.from_pretrained('bert-base-cased') 
          self.linear = torch.nn.Linear(self.bert.pooler.dense.out_features, 1)
          self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, input_ids, attention_mask, token_type_ids):
          # equivalent of having self.bert(**inputs)
          bert_output = self.bert(input_ids=input_ids, 
                                  attention_mask=attention_mask, 
                                  token_type_ids=token_type_ids)
          
          linear_output = self.linear(bert_output.last_hidden_state[0][0])

          return linear_output

