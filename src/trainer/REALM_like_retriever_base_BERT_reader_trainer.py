from data.medqa_questions import MedQAQuestions
from reader.reader import Reader
from retriever.retriever import Retriever
from trainer.trainer import Trainer
from transformers import get_linear_schedule_with_warmup
from data.data_loader import create_data_loader

import torch
import random
import numpy as np


class REALM_like_retriever_base_BERT_reader_trainer(Trainer):
    def __init__(self, questions: MedQAQuestions, retriever: Retriever, reader: Reader, num_epochs: int, batch_size: int, lr: float) -> None:
        super().__init__(questions, retriever, reader, num_epochs, batch_size, lr)

    def pepare_data_loader(self):
        print("******** Creating train dataloader ********")
        train_input_queries, train_input_answers, train_input_answers_idx = self.retriever.create_tokenized_input(
            questions=self.questions_train, tokenizer=self.reader.tokenizer, train_set=True)

        train_dataloader = create_data_loader(input_queries=train_input_queries, input_answers=train_input_answers,
                                              input_answers_idx=train_input_answers_idx, batch_size=self.batch_size)
        print("******** Train dataloader created  ********")

        print("******** Creating val dataloader ********")

        val_input_queries, val_input_answers, val_input_answers_idx = self.retriever.create_tokenized_input(
            questions=self.questions_val, tokenizer=self.reader.tokenizer, train_set=False)
        val_dataloader = create_data_loader(input_queries=val_input_queries, input_answers=val_input_answers,
                                            input_answers_idx=val_input_answers_idx, batch_size=self.batch_size)
        print("******** Val dataloader created  ********")

        return train_dataloader, val_dataloader

    def train(self):
        super().train()

        device = self.reader.device
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        training_stats = []

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.reader.model.parameters(), lr=self.lr)

        total_steps = self.num_epochs * self.batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        train_dataloader, val_dataloader = self.pepare_data_loader()
