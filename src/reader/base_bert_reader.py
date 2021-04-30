import torch
from reader.reader import Reader
from retriever.retriever import Retriever
from models.BaseBERTLinear import BaseBERTLinear
from transformers import AutoTokenizer

class Base_BERT_Reader(Reader):
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using {} device".format(self.device))
        
        self.model = BaseBERTLinear().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        
        
    def choose_answer(self, query, context, question_data):
        return "Nice"

    def create_context(self):
        return super().create_context()

    
   