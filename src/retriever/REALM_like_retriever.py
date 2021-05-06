import torch

from retriever.retriever import Retriever
from transformers import AutoTokenizer, AutoModel


class REALM_like_retriever(Retriever):
    tokenizer_type = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    d_encoder_bert_type = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    q_encoder_bert_type = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    # change to specify the weights file
    q_encoder_weights_path = ""

    def __init__(self, load_weights=False) -> None:
        super().__init__()
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using {} device".format(self.device))

        # defining tokenizer and encoders
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_type)
        self.d_encoder = AutoModel.from_pretrained(self.d_encoder_bert_type)
        self.q_encoder = AutoModel.from_pretrained(self.q_encoder_bert_type)

        # loading weights
        if load_weights:
            self.q_encoder.load_state_dict(
                torch.load(self.q_encoder_weights_path))

        # freezing layers
        self.freeze_layers(['pooler'])

    def retrieve_documents(self, query: str):
        return super().retrieve_documents(query)

    def freeze_layers(self, q_encoder_layers_to_not_freeze):
        for name, param in self.d_encoder.named_parameters():
            param.requires_grad = False

        for name, param in self.q_encoder.named_parameters():
            if not any(x in name for x in q_encoder_layers_to_not_freeze):
                param.requires_grad = False
            else:
                print(f"Layer {name} not frozen")
