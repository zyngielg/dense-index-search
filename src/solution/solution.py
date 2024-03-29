import datetime
import numpy as np
import random
import torch

from abc import ABC, abstractclassmethod
from data.medqa_questions import MedQAQuestions
from reader.reader import Reader
from numpy import sum, argmax
from retriever.retriever import Retriever


class Solution(ABC):
    def __init__(self, questions: MedQAQuestions, retriever: Retriever, reader: Reader, num_epochs: int, batch_size: int, lr: float) -> None:
        super().__init__()

        self.questions_train = questions.questions_train
        self.questions_val = questions.questions_val
        self.questions_test = questions.questions_test
        self.retriever = retriever
        self.reader = reader
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_answers = len(
            list(self.questions_train.values())[0]['options'])
        self.num_gpus = torch.cuda.device_count()

        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

    @abstractclassmethod
    def train(self):
        print("***** Running training *****")

    @abstractclassmethod
    def qa(self):
        print("***** Running QA *****")

    @staticmethod
    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    @staticmethod
    def calculate_accuracy(predictions_distribution, correct_answers, return_predictions = False):
        predictions = argmax(predictions_distribution, axis=1)
        if not return_predictions:
            return sum(predictions == correct_answers) / len(correct_answers)
        else:
            return sum(predictions == correct_answers) / len(correct_answers), predictions