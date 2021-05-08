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
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)
        self.freeze_layers()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.softmax = torch.nn.Softmax(dim=1)

    def choose_answer(self, query, context, question_data):
        raise NotImplementedError

    def create_context(self):
        return super().create_context()

    def freeze_layers(self):
        layers_to_not_freeze = ['10', '11', 'linear', 'pooler']  

        for name, param in self.model.named_parameters():
            if not any(x in name for x in layers_to_not_freeze):
                param.requires_grad = False
            else:
                print(f"Layer {name} not frozen")
