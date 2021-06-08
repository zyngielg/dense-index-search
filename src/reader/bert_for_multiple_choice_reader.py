import torch
from reader.reader import Reader
from transformers import AutoTokenizer, BertConfig, BertForMultipleChoice


class BERT_multiple_choice_reader(Reader):
    bert_name = "emilyalsentzer/Bio_ClinicalBERT"

    # change if necessary
    weights_file_directory = "src/trainer/results"
    weights_file_name = ""
    weights_file_path = f"{weights_file_directory}/{weights_file_name}"
    layers_to_not_freeze = ['8', '9', '10', '11', 'classifier', 'pooler']

    def __init__(self, load_weights=False):
        self.load_weights = load_weights
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using {} device".format(self.device))

        # bert_config = BertConfig.from_pretrained(bert_name)
        # self.model = BertForMultipleChoice(self.bert_config)
        self.model = BertForMultipleChoice.from_pretrained(self.bert_name)
        if load_weights:
            self.model.load_state_dict(torch.load(self.weights_file_path))
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
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
            else:
                print(
                    f"Layer {name} not frozen (status: {param.requires_grad})")

    def get_info(self):
        info = {}
        info['bert type'] = self.bert_name
        info['weights loaded'] = self.load_weights
        if self.load_weights:
            info['weights path'] = self.weights_file_path
        info['layers not to freeze'] = self.layers_to_not_freeze

        return info
