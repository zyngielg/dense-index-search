import datetime
import json
import numpy as np
import random
import time
import torch

from collections import Counter
from data.data_loader import create_questions_data_loader
from data.medqa_questions import MedQAQuestions
from retriever.colbert.colbert_retriever import ColBERTRetriever
from solution.solution import Solution
from transformers import get_linear_schedule_with_warmup
from utils.general_utils import remove_duplicates_preserve_order


class ColBERTe2e(Solution):
    def __init__(self, questions: MedQAQuestions, retriever: ColBERTRetriever, num_epochs: int, batch_size: int, lr: float) -> None:
        super().__init__(questions, retriever, None, num_epochs, batch_size, lr)
        self.num_train_questions = len(self.questions_train)
        self.num_val_questions = len(self.questions_val)
        self.num_test_questions = len(self.questions_test)

    def pepare_data_loader(self, include_test=False):
        print("******** Creating train dataloader ********")
        train_dataloader = create_questions_data_loader(
            questions=self.questions_train, batch_size=self.batch_size, num_questions=self.num_train_questions)
        print("******** Train dataloader created  ********")

        print("******** Creating val dataloader ********")
        val_dataloader = create_questions_data_loader(
            questions=self.questions_val, batch_size=self.batch_size, num_questions=self.num_val_questions)
        print("******** Val dataloader created  ********")

        if not include_test:
            return train_dataloader, val_dataloader
        else:
            test_dataloader = create_questions_data_loader(
                questions=self.questions_test, batch_size=self.batch_size, num_questions=self.num_test_questions)
            return train_dataloader, val_dataloader, test_dataloader

    def train(self):
        super().train()
        total_t0 = time.time()

        device = self.retriever.device
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        training_info = {
            "batch_size": self.batch_size,
            "lr": self.lr,
            "retriever": self.retriever.get_info(),
            "reader": "None - ColBERT e2e",
            "total_training_time": None,
            "num_train_questions": self.num_train_questions,
            "num_val_questions": self.num_val_questions,
            "training_stats": []
        }
        train_dataloader, val_dataloader = self.pepare_data_loader()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.retriever.colbert.parameters(), lr=self.lr)
        logsoftmax = torch.nn.LogSoftmax(dim=0)
        total_steps = self.num_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=1000,
                                                    num_training_steps=total_steps)

        for epoch in range(self.num_epochs):
            print(f'======== Epoch {epoch + 1} / {self.num_epochs} ========')
            t0 = time.time()
            total_train_loss = 0
            total_train_accuracy = 0
            self.retriever.colbert.train()
            for step, batch in enumerate(train_dataloader):
                if step % 50 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print(
                        f'Batch {step} of {len(train_dataloader)}. Elapsed: {elapsed}')
                self.retriever.colbert.zero_grad()
                questions = batch[0]
                answers_indexes = batch[2]
                options = [x.lower().split('#') for x in batch[3]]
                metamap_phrases = [x.split('#') for x in batch[4]]

                results = []
                for q_idx in range(len(questions)):

                    ### BEGINNING OF DOCUMENT RETRIEVAL ###
                    metamap_phrases[q_idx] = remove_duplicates_preserve_order(
                        metamap_phrases[q_idx])
                    query = ' '.join(metamap_phrases[q_idx])
                    # query = questions[q_idx]
                    query_options = [
                        x + f' {self.retriever.tokenizer.option_token} ' + query for x in options[q_idx]]
                    # query_options = [query + f' {self.retriever.tokenizer.option_token} ' + x for x in options[q_idx]]
                    # query_options = ['[unused5] ' + x + ' [unused6] ' + query for x in options[q_idx]]
                    retrieved_documents, scores = self.retriever.retrieve_documents(
                        query_options)
                    scores_mean = torch.mean((scores), dim=1)
                    # scores_mean = logsoftmax(scores_mean)
                    results.append(scores_mean)

                results = torch.stack(results)
                loss = criterion(results, answers_indexes.to("cuda:2"))
                if self.num_gpus > 1:
                    loss = loss.mean()
                total_train_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.retriever.colbert.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                # Move logits and labels to CPU
                if device.type == 'cpu':
                    output = results.numpy()
                    answers_indexes = answers_indexes.numpy()
                else:
                    output = results.detach().cpu().numpy()
                    answers_indexes = answers_indexes.to('cpu').numpy()
                total_train_accuracy += self.calculate_accuracy(
                    output, answers_indexes)

            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_train_accuracy = total_train_accuracy / len(train_dataloader)

            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.4f}".format(avg_train_loss))
            print("  Accuracy: {0:.4f}".format(avg_train_accuracy))
            print("  Training epoch took: {:}".format(training_time))

            print("Running Validation...")

            t0 = time.time()
            self.retriever.colbert.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0

            for step, batch in enumerate(val_dataloader):
                if step % 50 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print(
                        f'Batch {step} of {len(val_dataloader)}. Elapsed: {elapsed}')

                questions = batch[0]
                answers_indexes = batch[2]
                options = [x.lower().split('#') for x in batch[3]]
                metamap_phrases = [x.split('#') for x in batch[4]]

                results = []
                for q_idx in range(len(questions)):
                    with torch.no_grad():
                        metamap_phrases[q_idx] = remove_duplicates_preserve_order(
                            metamap_phrases[q_idx])
                        query = ' '.join(metamap_phrases[q_idx])
                        # query = questions[q_idx]
                        query_options = [
                            x + f' {self.retriever.tokenizer.option_token} ' + query for x in options[q_idx]]
                        # query_options = ['[unused5] ' + x + ' [unused6] ' + query for x in options[q_idx]]
                        retrieved_documents, scores = self.retriever.retrieve_documents(
                            query_options)
                        scores_mean = torch.mean((scores), dim=1)
                        # scores_mean = logsoftmax(scores_mean)
                        results.append(scores_mean)

                results = torch.stack(results)
                loss = criterion(results, answers_indexes.to("cuda:2"))
                if self.num_gpus > 1:
                    loss = loss.mean()
                total_eval_loss += loss.item()

                if device.type == 'cpu':
                    output = results.numpy()
                    answers_indexes = answers_indexes.numpy()
                else:
                    output = results.detach().cpu().numpy()
                    answers_indexes = answers_indexes.to('cpu').numpy()
                total_eval_accuracy += self.calculate_accuracy(
                    output, answers_indexes)

            avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
            avg_val_loss = total_eval_loss / len(val_dataloader)

            validation_time = self.format_time(time.time() - t0)

            print("  Validation Loss: {0:.4f}".format(avg_val_loss))
            print("  Accuracy: {0:.4f}".format(avg_val_accuracy))
            print("  Validation took: {:}".format(validation_time))

            training_info['training_stats'].append(
                {
                    'epoch': epoch + 1,
                    'Training Time': training_time,
                    'Training Loss': avg_train_loss,
                    'Training Accuracy.': avg_train_accuracy,
                    'Validation Time': validation_time,
                    'Validation Loss': avg_val_loss,
                    'Validation Accuracy.': avg_val_accuracy
                }
            )

        print("Training complete!")

        total_training_time = self.format_time(time.time()-total_t0)
        training_info['total_training_time'] = total_training_time
        print(f"Total training took {training_time} (h:mm:ss)")

        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
        # saving training stats
        training_stats_file = f"src/results/colbert-based/{dt_string}__ColBERT_e2e_stats.json"
        with open(training_stats_file, 'w') as results_file:
            json.dump(training_info, results_file)
        print(f"Results saved in {training_stats_file}")

        colbert_file_name = f"src/results/colbert-based/{dt_string}__ColBERT_e2e_retriever.pth"
        torch.save(self.retriever.colbert.state_dict(), colbert_file_name)
        print(f"Reader weights saved in {colbert_file_name}")

        print("***** Training completed *****")

    def qa(self):
        total_t0 = time.time()
        qa_info = {
            "batch_size": self.batch_size,
            "lr": self.lr,
            "retriever": self.retriever.get_info(),
            "reader": "None - ColBERT e2e",
            "total_training_time": None,
            "num_train_questions": self.num_train_questions,
            "num_val_questions": self.num_val_questions,
            "training_stats": []
        }
        self.retriever.colbert.eval()
        train_dataloader, val_dataloader, test_dataloader = self.pepare_data_loader(
            include_test=True)

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
        qa_stats_file = f"src/results/ir-es-based/{dt_string}__QA__ColBERT_e2e.json"
        with open(qa_stats_file, 'w') as results_file:
            json.dump(qa_info, results_file)
        print("********* QA complete *********")

    def __question_answering(self, data_loader):
        total_accuracy = 0
        t0 = time.time()
        all_predictions = []
        device = self.retriever.device

        for step, batch in enumerate(data_loader):
            if step % 25 == 0 and not step == 0:
                elapsed = self.format_time(time.time() - t0)
                print(
                    f'Batch {step} of {len(data_loader)}. Elapsed: {elapsed}')
            questions = batch[0]
            answers_indexes = batch[2]
            options = [x.lower().split('#') for x in batch[3]]
            metamap_phrases = [x.split('#') for x in batch[4]]

            results = []
            for q_idx in range(len(questions)):
                metamap_phrases[q_idx] = remove_duplicates_preserve_order(
                    metamap_phrases[q_idx])
                query = ' '.join(metamap_phrases[q_idx])
                query_options = [
                    x + f' {self.retriever.tokenizer.option_token} ' + query for x in options[q_idx]]
                retrieved_documents, scores = self.retriever.retrieve_documents(
                    query_options)
                scores_mean = torch.mean((scores), dim=1)
                results.append(scores_mean)

            results = torch.stack(results)
            if device.type == 'cpu':
                output = results.numpy()
                answers_indexes = answers_indexes.numpy()
            else:
                output = results.detach().cpu().numpy()
                answers_indexes = answers_indexes.to('cpu').numpy()
            accuracy, predictions = self.calculate_accuracy(
                output, answers_indexes, True)
            total_accuracy += accuracy
            all_predictions += [int(x) for x in list(predictions)]

        avg_accuracy = total_accuracy / len(data_loader)
        qa_time = self.format_time(time.time() - t0)

        print(f"  Time: {qa_time}")
        print("  Accuracy: {0:.4f}".format(avg_accuracy))
        return avg_accuracy, all_predictions
