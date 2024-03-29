import datetime
import json
import numpy as np
import random
import time
import torch
from torch.utils.data import dataloader

from collections import Counter
from data.data_loader import create_medqa_data_loader
from data.medqa_questions import MedQAQuestions
from reader.reader import Reader
from retriever.retriever import Retriever
from solution.solution import Solution
from transformers import get_linear_schedule_with_warmup


class IrEsBaseBert(Solution):
    def __init__(self, questions: MedQAQuestions, retriever: Retriever, reader: Reader, num_epochs: int, batch_size: int, lr: float) -> None:
        super().__init__(questions, retriever, reader, num_epochs, batch_size, lr)

    def pepare_data_loader(self, include_test=False):
        print("********* Creating train dataloader ... *********")
        train_input_queries, train_input_answers, train_input_answers_idx = self.retriever.create_tokenized_input(
            questions=self.questions_train,
            tokenizer=self.reader.tokenizer,
            docs_flag=0,
            num_questions=len(self.questions_train),
            medqa=False)

        train_dataloader = create_medqa_data_loader(input_queries=train_input_queries, input_answers=train_input_answers,
                                                    input_answers_idx=train_input_answers_idx, batch_size=self.batch_size)
        print("********* ... train dataloader created  *********")

        print("********* Creating val dataloader ... *********")

        val_input_queries, val_input_answers, val_input_answers_idx = self.retriever.create_tokenized_input(
            questions=self.questions_val,
            tokenizer=self.reader.tokenizer,
            docs_flag=1,
            num_questions=len(self.questions_val),
            medqa=False)
        val_dataloader = create_medqa_data_loader(input_queries=val_input_queries, input_answers=val_input_answers,
                                                  input_answers_idx=val_input_answers_idx, batch_size=self.batch_size)
        print("********* ... val dataloader created  *********")
        if not include_test:
            return train_dataloader, val_dataloader
        else:
            test_input_queries, test_input_answers, test_input_answers_idx = self.retriever.create_tokenized_input(
                questions=self.questions_test,
                tokenizer=self.reader.tokenizer,
                docs_flag=2,
                num_questions=len(self.questions_test),
                medqa=False)
            test_dataloader = create_medqa_data_loader(input_queries=test_input_queries, input_answers=test_input_answers,
                                                       input_answers_idx=test_input_answers_idx, batch_size=self.batch_size)
            return train_dataloader, val_dataloader, test_dataloader

    def train(self):
        super().train()
        device = self.reader.device
        total_t0 = time.time()

        training_info = {
            "retriever": self.retriever.get_info(),
            "reader": self.reader.get_info(),
            "total_training_time": None,
            "training_stats": []
        }
        num_epochs = self.num_epochs
        train_dataloader, val_dataloader = self.pepare_data_loader()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.reader.model.parameters(), lr=self.lr)

        total_steps = self.num_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=1000,
                                                    num_training_steps=total_steps)

        for epoch in range(num_epochs):
            print(f'========= Epoch {epoch + 1} / {self.num_epochs} =========')
            t0 = time.time()
            print("****** Training ******")
            self.reader.model.train()
            total_train_loss = 0
            total_train_accuracy = 0
            for step, batch in enumerate(train_dataloader):
                if step % 25 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print(
                        f'Batch {step} of {len(train_dataloader)}. Elapsed: {elapsed}')

                optimizer.zero_grad()  # no difference if model or optimizer.zero_grad

                questions_queries_collection = batch[0]
                answers_indexes = batch[2]

                # get the tensors
                input_ids = questions_queries_collection["input_ids"]
                input_token_type_ids = questions_queries_collection["token_type_ids"]
                input_attention_mask = questions_queries_collection["attention_mask"]

                output = self.reader.model(input_ids=input_ids,
                                           attention_mask=input_attention_mask,
                                           token_type_ids=input_token_type_ids)

                loss = criterion(output, answers_indexes.to(device))
                if self.num_gpus > 1:
                    loss = loss.mean()
                total_train_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.reader.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                if device.type == 'cpu':
                    output = output.numpy()
                    answers_indexes = answers_indexes.numpy()
                else:
                    output = output.detach().cpu().numpy()
                    answers_indexes = answers_indexes.to('cpu').numpy()
                total_train_accuracy += self.calculate_accuracy(
                    output, answers_indexes)

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_train_accuracy = total_train_accuracy / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            print("\tAverage training loss: {0:.4f}".format(avg_train_loss))
            print("  Accuracy: {0:.4f}".format(avg_train_accuracy))
            print("\tTraining epoch took: {:}".format(training_time))

            print("****** Validation ******")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.reader.model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0

            # Evaluate data for one epoch
            for step, batch in enumerate(val_dataloader):
                questions_queries_collection = batch[0]
                answers_indexes = batch[2]

                # get the tensors
                input_ids = questions_queries_collection["input_ids"]
                input_token_type_ids = questions_queries_collection["token_type_ids"]
                input_attention_mask = questions_queries_collection["attention_mask"]

                with torch.no_grad():
                    output = self.reader.model(input_ids=input_ids.to(device),
                                               attention_mask=input_attention_mask.to(
                                                   device),
                                               token_type_ids=input_token_type_ids.to(device))

                loss = criterion(output, answers_indexes.to(device))

                if self.num_gpus > 1:
                    loss = loss.mean()
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                if device.type == 'cpu':
                    output = output.numpy()
                    answers_indexes = answers_indexes.numpy()
                else:
                    output = output.detach().cpu().numpy()
                    answers_indexes = answers_indexes.to('cpu').numpy()
                total_eval_accuracy += self.calculate_accuracy(
                    output, answers_indexes)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
            print("\tAccuracy: {0:.4f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(val_dataloader)

            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)

            print("  Validation Loss: {0:.4f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_info['training_stats'].append(
                {
                    'epoch': epoch + 1,
                    'Training Loss': avg_train_loss,
                    'Training Accuracy': avg_train_accuracy,
                    'Validation Loss': avg_val_loss,
                    'Validation Accuracy': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("********* Training complete *********")
        total_training_time = self.format_time(time.time()-total_t0)
        training_info['total_training_time'] = total_training_time
        print(f"Total training took {training_time} (h:mm:ss)")

        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
        # saving training stats
        training_stats_file = f"src/results/ir-es-based/{dt_string}__IR-ES__base-BERT.json"
        with open(training_stats_file, 'w') as results_file:
            json.dump(training_info, results_file)
        # saving the reader weights
        reader_file_name = f"src/results/ir-es-based/{dt_string}__IR-ES__base-BERT.pth"
        torch.save(self.reader.model.state_dict(), reader_file_name)
        print(f"Reader weights saved in {reader_file_name}")
        print("***** Training completed *****")

    def qa(self):
        total_t0 = time.time()
        device = self.reader.device
        qa_info = {
            "retriever": self.retriever.get_info(),
            "reader": self.reader.get_info(),
            "total_training_time": None,
            "training_stats": []
        }
        self.reader.model.eval()
        train_dataloader, val_dataloader, test_dataloader = self.pepare_data_loader(
            True)

        print("*** QA for training set")
        train_accuracy, train_predictions = self.__question_answering(
            train_dataloader, device)
        qa_info["train_accuracy"] = train_accuracy
        qa_info["train_predictions"] = train_predictions

        print("*** QA for val set")
        val_accuracy, val_predictions = self.__question_answering(
            val_dataloader, device)
        qa_info["val_accuracy"] = val_accuracy
        qa_info["val_predictions"] = val_predictions

        print("*** QA for test set")
        test_accuracy, test_predictions = self.__question_answering(
            test_dataloader, device)
        qa_info["test_accuracy"] = test_accuracy
        qa_info["test_predictions"] = test_predictions

        all_predictions = train_predictions + val_predictions + test_predictions
        predictions_dist = sorted(Counter(all_predictions).items())
        qa_info["all_predicitions_dist"] = predictions_dist

        total_time = self.format_time(time.time()-total_t0)
        qa_info['total_qa_time'] = total_time
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
        qa_stats_file = f"src/results/ir-es-based/{dt_string}__QA__IR-ES__base-BERT.json"
        with open(qa_stats_file, 'w') as results_file:
            json.dump(qa_info, results_file)
        print("********* QA complete *********")

    def __question_answering(self, data_loader, device):
        total_accuracy = 0
        t0 = time.time()
        all_predictions = []
        for step, batch in enumerate(data_loader):
            if step % 25 == 0 and not step == 0:
                elapsed = self.format_time(time.time() - t0)
                print(
                    f'Batch {step} of {len(data_loader)}. Elapsed: {elapsed}')

            questions_queries_collection = batch[0]
            answers_indexes = batch[2]

            input_ids = questions_queries_collection["input_ids"]
            input_token_type_ids = questions_queries_collection["token_type_ids"]
            input_attention_mask = questions_queries_collection["attention_mask"]

            output = self.reader.model(input_ids=input_ids,
                                       attention_mask=input_attention_mask,
                                       token_type_ids=input_token_type_ids)

            if device.type == 'cpu':
                output = output.numpy()
                answers_indexes = answers_indexes.numpy()
            else:
                output = output.detach().cpu().numpy()
                answers_indexes = answers_indexes.to('cpu').numpy()
            accuracy, predictions = self.calculate_accuracy(
                output, answers_indexes, return_predictions=True)
            total_accuracy += accuracy
            all_predictions += [int(x) for x in list(predictions)]

        avg_accuracy = total_accuracy / len(data_loader)
        qa_time = self.format_time(time.time() - t0)

        print(f"  Time: {qa_time}")
        print("  Accuracy: {0:.4f}".format(avg_accuracy))
        return avg_accuracy, all_predictions
