import datetime
import json
import time
import torch

from collections import Counter
from data.data_loader import create_questions_data_loader
from data.medqa_questions import MedQAQuestions
from reader.reader import Reader
from retriever.base_bert_retriever import BaseBertRetriever
from solution.solution import Solution
from transformers import get_linear_schedule_with_warmup
from utils.general_utils import remove_duplicates_preserve_order


class REALMLikeRetrieverBaseBERTReader(Solution):
    def __init__(self, questions: MedQAQuestions, retriever: BaseBertRetriever, reader: Reader, num_epochs: int, batch_size: int, lr: float) -> None:
        super().__init__(questions, retriever, reader, num_epochs, batch_size, lr)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def pepare_data_loader(self, include_test=False):
        print("******** Creating train dataloader ********")
        train_dataloader = create_questions_data_loader(
            questions=self.questions_train,
            batch_size=self.batch_size,
            num_questions=len(self.questions_train))
        print("******** Train dataloader created  ********")

        print("******** Creating val dataloader ********")
        val_dataloader = create_questions_data_loader(
            questions=self.questions_val, batch_size=self.batch_size, num_questions=len(self.questions_val))
        print("******** Val dataloader created  ********")

        if not include_test:
            return train_dataloader, val_dataloader
        else:
            test_dataloader = create_questions_data_loader(
                questions=self.questions_test, batch_size=self.batch_size, num_questions=len(self.questions_test))
            return train_dataloader, val_dataloader, test_dataloader

    def train(self):
        super().train()
        total_t0 = time.time()
        device = self.reader.device

        training_info = {
            "batch_size": self.batch_size,
            "lr": self.lr,
            "retriever": self.retriever.get_info(),
            "reader": self.reader.get_info(),
            "total_training_time": None,
            "training_stats": []
        }
        train_dataloader, val_dataloader = self.pepare_data_loader()
        criterion = torch.nn.CrossEntropyLoss()
        params = list(self.retriever.q_embedder.parameters()) + \
            list(self.reader.model.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr)

        total_steps = self.num_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=1000,
                                                    num_training_steps=total_steps)

        self.retriever.d_embedder.eval()
        for epoch in range(self.num_epochs):
            print(f'======== Epoch {epoch + 1} / {self.num_epochs} ========')
            t0 = time.time()
            total_train_loss = 0
            total_train_accuracy = 0
            self.retriever.q_embedder.train()
            self.reader.model.train()
            for step, batch in enumerate(train_dataloader):
                if step % 10 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print(
                        f'Batch {step} of {len(train_dataloader)}. Elapsed: {elapsed}')
                optimizer.zero_grad()
                self.retriever.q_embedder.zero_grad()
                questions = batch[0]
                answers_indexes = batch[2]
                options = [x.split('#') for x in batch[3]]
                metamap_phrases = [x.split('#') for x in batch[4]]

                input_ids = []
                token_type_ids = []
                attention_masks = []
                # retriever_scores = []
                for q_idx in range(len(questions)):
                    metamap_phrases[q_idx] = remove_duplicates_preserve_order(
                        metamap_phrases[q_idx])
                    query = ' '.join(metamap_phrases[q_idx])
                    query_options = ['[unused5] ' + x +
                                     ' [unused6] ' + query for x in options[q_idx]]
                    scores, retrieved_documents = self.retriever.retrieve_documents(
                        query_options)

                    contexts = []
                    for idx in range(len(retrieved_documents)):
                        option_documents = []
                        for document in retrieved_documents[idx]:
                            option_documents.append(document)
                        contexts.append(' '.join(option_documents))

                    # retriever_scores.append(torch.mean(scores, dim=1))
                    question_inputs = self.reader.tokenizer(
                        contexts, query_options, add_special_tokens=True, max_length=512, padding='max_length', truncation='longest_first', return_tensors="pt")
                    input_ids.append(question_inputs['input_ids'])
                    token_type_ids.append(question_inputs['token_type_ids'])
                    attention_masks.append(question_inputs['attention_mask'])

                tensor_input_ids = torch.stack(
                    input_ids, dim=0)  # .to(device="cuda:7")
                tensor_token_type_ids = torch.stack(
                    token_type_ids, dim=0)  # .to(device="cuda:7")
                tensor_attention_masks = torch.stack(
                    attention_masks, dim=0)  # .to(device="cuda:7")
                # retriever_scores = torch.stack(retriever_scores, dim=0)
                output = self.reader.model(
                    input_ids=tensor_input_ids, attention_mask=tensor_attention_masks, token_type_ids=tensor_token_type_ids)

                # retriever_score = 0 # log_softmax(retriever_scores)
                reader_score = self.log_softmax(output)
                sum_score = reader_score  # + retriever_score
                loss = criterion(sum_score, answers_indexes.to(device))
                if self.num_gpus > 1:
                    loss = loss.mean()
                total_train_loss += loss.item()
                loss.backward()
                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(
                    self.reader.model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(
                    self.retriever.q_embedder.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                if device.type == 'cpu':
                    output = sum_score.numpy()
                    answers_indexes = answers_indexes.numpy()
                else:
                    output = sum_score.detach().cpu().numpy()
                    answers_indexes = answers_indexes.to('cpu').numpy()
                total_train_accuracy += self.calculate_accuracy(
                    output, answers_indexes)

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_train_acc = total_train_accuracy / len(train_dataloader)
            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.4f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            t0 = time.time()
            self.retriever.q_embedder.eval()
            self.reader.model.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0

            for step, batch in enumerate(val_dataloader):
                if step % 25 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print(
                        f'Batch {step} of {len(val_dataloader)}. Elapsed: {elapsed}')
                questions = batch[0]
                answers_indexes = batch[2]
                options = [x.split('#') for x in batch[3]]
                metamap_phrases = [x.split('#') for x in batch[4]]

                input_ids = []
                token_type_ids = []
                attention_masks = []
                # retriever_scores = []
                for q_idx in range(len(questions)):
                    metamap_phrases[q_idx] = remove_duplicates_preserve_order(
                        metamap_phrases[q_idx])
                    query = ' '.join(metamap_phrases[q_idx])
                    query_options = ['[unused5] ' + x +
                                     ' [unused6] ' + query for x in options[q_idx]]
                    with torch.no_grad():
                        scores, retrieved_documents = self.retriever.retrieve_documents(
                            query_options)

                    contexts = []
                    for idx in range(len(retrieved_documents)):
                        option_documents = []
                        for document in retrieved_documents[idx]:
                            option_documents.append(document)
                        contexts.append(' '.join(option_documents))

                    # retriever_scores.append(torch.mean(scores, dim=1))
                    question_inputs = self.reader.tokenizer(contexts, query_options,
                                                            add_special_tokens=True,
                                                            max_length=512,
                                                            padding='max_length',
                                                            truncation='longest_first',
                                                            return_tensors="pt")
                    input_ids.append(question_inputs['input_ids'])
                    token_type_ids.append(question_inputs['token_type_ids'])
                    attention_masks.append(question_inputs['attention_mask'])

                tensor_input_ids = torch.stack(input_ids, dim=0)
                tensor_token_type_ids = torch.stack(token_type_ids, dim=0)
                tensor_attention_masks = torch.stack(attention_masks, dim=0)
                # retriever_scores = torch.stack(retriever_scores, dim=0)

                with torch.no_grad():
                    output = self.reader.model(
                        input_ids=tensor_input_ids.to(device), attention_mask=tensor_token_type_ids.to(device), token_type_ids=tensor_attention_masks.to(device))

                # retriever_score = 0 #log_softmax(retriever_scores)
                reader_score = self.log_softmax(output)
                sum_score = reader_score  # +  retriever_score
                loss = criterion(sum_score, answers_indexes.to(device))
                if self.num_gpus > 1:
                    loss = loss.mean()
                total_eval_loss += loss.item()

                if device.type == 'cpu':
                    output = sum_score.numpy()
                    answers_indexes = answers_indexes.numpy()
                else:
                    output = sum_score.detach().cpu().numpy()
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
                    'Training Loss': avg_train_loss,
                    'Training Accuracy': avg_train_acc,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("Training complete!")

        total_training_time = self.format_time(time.time()-total_t0)
        training_info['total_training_time'] = total_training_time
        print(f"Total training took {training_time} (h:mm:ss)")

        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
        # saving training stats
        training_stats_file = f"src/results/realm-based/{dt_string}__REALM_retriever__base_BERT_reader.json"
        with open(training_stats_file, 'w') as results_file:
            json.dump(training_info, results_file)
        print(f"Results saved in {training_stats_file}")
        # saving the retriever's q_embedder weights
        retriever_file_name = f"src/results/realm-based/{dt_string}__REALM_retriever.pth"
        torch.save(self.retriever.q_embedder.state_dict(), retriever_file_name)
        print(f"Q_encoder weights saved in {retriever_file_name}")
        # saving the reader weights
        reader_file_name = f"src/results/realm-based/{dt_string}__BERT_reader.pth"
        torch.save(self.reader.model.state_dict(), reader_file_name)
        print(f"Reader weights saved in {retriever_file_name}")

        print("***** Training completed *****")

    def qa(self):
        total_t0 = time.time()
        qa_info = {
            "batch_size": self.batch_size,
            "lr": self.lr,
            "retriever": self.retriever.get_info(),
            "reader": self.reader.get_info(),
            "total_training_time": None,
            "training_stats": []
        }
        self.retriever.d_embedder.eval()
        train_dataloader, val_dataloader, test_dataloader = self.pepare_data_loader(True)

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
        qa_stats_file = f"src/results/realm-based/{dt_string}__QA__REALM_retriever__base_BERT_reader.json"
        with open(qa_stats_file, 'w') as results_file:
            json.dump(qa_info, results_file)
        print("********* QA complete *********")

    def __question_answering(self, data_loader):
        device = self.reader.device
        total_accuracy = 0
        t0 = time.time()
        all_predictions = []

        for step, batch in enumerate(data_loader):
            if step % 10 == 0 and not step == 0:
                elapsed = self.format_time(time.time() - t0)
                print(
                    f'Batch {step} of {len(data_loader)}. Elapsed: {elapsed}')
            questions = batch[0]
            answers_indexes = batch[2]
            options = [x.split('#') for x in batch[3]]
            metamap_phrases = [x.split('#') for x in batch[4]]

            input_ids = []
            token_type_ids = []
            attention_masks = []
            # retriever_scores = []
            for q_idx in range(len(questions)):
                metamap_phrases[q_idx] = remove_duplicates_preserve_order(
                    metamap_phrases[q_idx])
                query = ' '.join(metamap_phrases[q_idx])
                query_options = ['[unused5] ' + x +
                                 ' [unused6] ' + query for x in options[q_idx]]
                scores, retrieved_documents = self.retriever.retrieve_documents(
                    query_options)

                contexts = []
                for idx in range(len(retrieved_documents)):
                    option_documents = []
                    for document in retrieved_documents[idx]:
                        option_documents.append(document)
                    contexts.append(' '.join(option_documents))

                # retriever_scores.append(torch.mean(scores, dim=1))
                question_inputs = self.reader.tokenizer(
                    contexts, query_options, add_special_tokens=True, max_length=512, padding='max_length', truncation='longest_first', return_tensors="pt")
                input_ids.append(question_inputs['input_ids'])
                token_type_ids.append(question_inputs['token_type_ids'])
                attention_masks.append(question_inputs['attention_mask'])

            tensor_input_ids = torch.stack(input_ids, dim=0)
            tensor_token_type_ids = torch.stack(token_type_ids, dim=0)
            tensor_attention_masks = torch.stack(attention_masks, dim=0)
            # retriever_scores = torch.stack(retriever_scores, dim=0)
            output = self.reader.model(
                input_ids=tensor_input_ids, attention_mask=tensor_attention_masks, token_type_ids=tensor_token_type_ids)

            # retriever_score = 0 # log_softmax(retriever_scores)
            reader_score = self.log_softmax(output)
            sum_score = reader_score  # + retriever_score

            if device.type == 'cpu':
                output = sum_score.numpy()
                answers_indexes = answers_indexes.numpy()
            else:
                output = sum_score.detach().cpu().numpy()
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
