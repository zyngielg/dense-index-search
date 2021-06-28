import torch
from reader.reader import Reader
from retriever.retriever import Retriever
from models.BaseBERTLinear import BaseBERTLinear
from transformers import AutoTokenizer


class Base_BERT_Reader(Reader):
    # bert_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    bert_name = "emilyalsentzer/Bio_ClinicalBERT"
    # bert_name = "bert-base-uncased"
    # change if necessary
    weights_file_directory = "src/results/ir-es-based"
    weights_file_name = "2021-06-22_18:17:21__IR-ES__base-BERT.pth"
    weights_file_path = f"{weights_file_directory}/{weights_file_name}"
    layers_to_not_freeze = ['7', '8', '9', '10', '11', 'linear', 'pooler']

    def __init__(self, load_weights=False):
        self.load_weights = load_weights
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using {} device".format(self.device))

        self.model = BaseBERTLinear(self.bert_name)
        if load_weights:
            saved_model = torch.load(self.weights_file_path)
            saved_model = {key.replace("module.", ""): value for key, value in saved_model.items()}
            self.model.load_state_dict(saved_model)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            if torch.cuda.device_count() == 8:
                self.model = torch.nn.DataParallel(
                    self.model, device_ids=[7, 6, 5])
                self.device = torch.device('cuda:7')
            else:
                self.model = torch.nn.DataParallel(self.model)

        self.freeze_layers()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name)
        self.softmax = torch.nn.Softmax(dim=1)

    def choose_answer(self, query, context, question_data):
        raise NotImplementedError

    def create_context(self):
        return super().create_context()

    def freeze_layers(self):
        for name, param in self.model.named_parameters():
            if not any(x in name for x in self.layers_to_not_freeze):
                param.requires_grad = False
            # else:
            #     print(f"Layer {name} not frozen (status: {param.requires_grad})")

    def get_info(self):
        info = {}
        info['bert type'] = self.bert_name
        info['weights loaded'] = self.load_weights
        if self.load_weights:
            info['weights path'] = self.weights_file_path
        info['layers not to freeze'] = self.layers_to_not_freeze

        return info
