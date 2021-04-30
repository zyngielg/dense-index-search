import torch
from reader.reader import Reader
from retriever.retriever import Retriever
from models.BaseBERTLinear import BaseBERTLinear
from transformers import AutoTokenizer


class Base_BERT_Reader(Reader):
    # change if necessary
    weights_file_directory = "src/trainer/results"
    weights_file_name = "2021-04-30_16:08:19 reader IRES retriever BERT_linear.pth"
    weights_file_path = f"{weights_file_directory}/{weights_file_name}"

    def __init__(self, load_weights=False):
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using {} device".format(self.device))

        self.model = BaseBERTLinear()
        if load_weights:
            self.model.load_state_dict(torch.load(self.weights_file_path))

        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def choose_answer(self, query, context, question_data):
        return "Nice"

    def create_context(self):
        return super().create_context()
