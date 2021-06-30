import json
import datetime
import numpy as np
import time

from collections import Counter
from data.data_loader import create_questions_data_loader
from data.medqa_questions import MedQAQuestions
from random import randint
from reader.reader import Reader
from retriever.retriever import Retriever
from solution.solution import Solution


class RandomGuesser(Solution):
    def __init__(self, questions: MedQAQuestions, retriever: Retriever, reader: Reader, num_epochs: int, batch_size: int, lr: float) -> None:
        super().__init__(questions, retriever, reader, num_epochs, batch_size, lr)

    def qa(self):
        total_t0 = time.time()
        train_dataloader, val_dataloader, test_dataloader = self.__pepare_data_loader()

        qa_info = {}

        print("*** QA for training set")
        train_accuracy, train_predictions = self.__question_answering(
            train_dataloader)
        qa_info["train_accuracy"] = train_accuracy
        qa_info["train_predictions"] = train_predictions

        print("*** QA for val set")
        val_accuracy, val_predictions = self.__question_answering(
            val_dataloader)
        qa_info["val_accuracy"] = val_accuracy
        qa_info["val_predictions"] = val_predictions

        print("*** QA for test set")
        test_accuracy, test_predictions = self.__question_answering(
            test_dataloader)
        qa_info["test_accuracy"] = test_accuracy
        qa_info["test_predictions"] = test_predictions

        all_predictions = train_predictions + val_predictions + test_predictions
        predictions_dist = sorted(Counter(all_predictions).items())
        qa_info["all_predicitions_dist"] = predictions_dist

        total_time = self.format_time(time.time()-total_t0)
        qa_info['total_qa_time'] = total_time
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
        qa_stats_file = f"src/results/{dt_string}__random_guesser.json"
        with open(qa_stats_file, 'w') as results_file:
            json.dump(qa_info, results_file)
        print("********* QA complete *********")

    def train(self):
        print("Random guesser is not a trainable solution.")
        quit()

    def __question_answering(self, data_loader):
        all_predictions = []

        for step, batch in enumerate(data_loader):
            questions = batch[0]
            answers_indexes = batch[2]
            for question in questions:
                chosen_answer = randint(0, 3)
                all_predictions.append(chosen_answer)

        accuracy = sum(np.array(all_predictions) == np.array(
            answers_indexes)) / len(answers_indexes)
        print("  Accuracy: {0:.4f}".format(accuracy))
        return accuracy, all_predictions

    def __pepare_data_loader(self):
        train_dataloader = create_questions_data_loader(questions=self.questions_train,
                                                        batch_size=len(
                                                            self.questions_train),
                                                        num_questions=len(self.questions_train))

        val_dataloader = create_questions_data_loader(questions=self.questions_val,
                                                      batch_size=len(
                                                          self.questions_val),
                                                      num_questions=len(self.questions_val))

        test_dataloader = create_questions_data_loader(questions=self.questions_test,
                                                       batch_size=len(
                                                           self.questions_test),
                                                       num_questions=len(self.questions_test))
        return train_dataloader, val_dataloader, test_dataloader
