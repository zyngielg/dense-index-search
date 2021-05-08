import datetime

from abc import ABC, abstractclassmethod
from data.medqa_questions import MedQAQuestions
from reader.reader import Reader
from numpy import sum, argmax
from retriever.retriever import Retriever


class Trainer(ABC):
    def __init__(self, questions: MedQAQuestions, retriever: Retriever, reader: Reader, num_epochs: int, batch_size: int, lr: float) -> None:
        super().__init__()

        self.questions_train = questions.questions_train
        self.questions_val = questions.questions_val
        self.retriever = retriever
        self.reader = reader
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_answers = len(
            list(self.questions_train.values())[0]['options'])

    @abstractclassmethod
    def train(self):
        print("***** Running training *****")

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
    def calculate_accuracy(predictions_distribution, correct_answers):
        predictions = argmax(predictions_distribution, axis=1)
        return sum(predictions == correct_answers) / len(correct_answers)
