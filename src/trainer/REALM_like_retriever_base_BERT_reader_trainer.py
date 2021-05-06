from data.medqa_questions import MedQAQuestions
from reader.reader import Reader
from retriever.retriever import Retriever
from trainer.trainer import Trainer


class REALM_like_retriever_base_BERT_reader_trainer(Trainer):
    def __init__(self, questions: MedQAQuestions, retriever: Retriever, reader: Reader, num_epochs: int, batch_size: int, lr: float) -> None:
        super().__init__(questions, retriever, reader, num_epochs, batch_size, lr)
        
    def train(self):
        super().train()
        print("Training whohoo")