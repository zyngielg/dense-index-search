import datetime
import json
import numpy as np
import random
import time
import torch

from data.data_loader import create_questions_data_loader
from data.medqa_questions import MedQAQuestions
from reader.bert_for_multiple_choice_reader import BERT_multiple_choice_reader
from retriever.base_bert_retriever import BaseBertRetriever
from solution.solution import Solution
from transformers import get_linear_schedule_with_warmup


class BaseBERTRetrieverBERTForMultipleChoiceReader(Solution):
    def __init__(self, questions: MedQAQuestions, retriever: BaseBertRetriever, reader: BERT_multiple_choice_reader, num_epochs: int, batch_size: int, lr: float) -> None:
        super().__init__(questions, retriever, reader, num_epochs, batch_size, lr)

    def pepare_data_loader(self):
        print("******** Creating train dataloader ********")
        train_dataloader = create_questions_data_loader(
            questions=self.questions_train, tokenizer=self.retriever.tokenizer, batch_size=self.batch_size)
        print("******** Train dataloader created  ********")

        print("******** Creating val dataloader ********")
        val_dataloader = create_questions_data_loader(
            questions=self.questions_val, tokenizer=self.retriever.tokenizer, batch_size=self.batch_size)
        print("******** Val dataloader created  ********")

        return train_dataloader, val_dataloader

    def train(self):
        super().train()
        total_t0 = time.time()

        device = self.reader.device
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        training_info = {
            "retriever": self.retriever.get_info(),
            "reader": self.reader.get_info(),
            "tokenizer": self.retriever.tokenizer_type,
            "total_training_time": None,
            "training_stats": []
        }

        params = list(self.retriever.q_encoder.parameters()) + \
            list(self.reader.model.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr)

        total_steps = self.num_epochs * self.batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        train_dataloader, val_dataloader = self.pepare_data_loader()

        for epoch in range(self.num_epochs):
            print(f'======== Epoch {epoch + 1} / {self.num_epochs} ========')
            t0 = time.time()
            total_train_loss = 0
            self.retriever.q_encoder.train()
            self.reader.model.train()
            for step, batch in enumerate(train_dataloader):
                if step % 25 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print(
                        f'Batch {step} of {len(train_dataloader)}. Elapsed: {elapsed}')

                optimizer.zero_grad()
                # self.reader.model.zero_grad()
                # self.retriever.q_encoder.zero_grad()

                questions = batch[0]
                answers_indexes = batch[2]
                options = batch[3]
                metamap_phrases = batch[4]

                input_ids = []
                token_type_ids = []
                attention_masks = []

                for q_idx in range(len(questions)):
                    query = ' '.join(metamap_phrases[q_idx])
                    query_options = [query + ' ' + x[q_idx] for x in options]
                    retrieved_documents = [
                        self.retriever.retrieve_documents(x) for x in query_options]

                    contexts = []
                    for idx in range(len(retrieved_documents)):
                        option_documents = []
                        for document in retrieved_documents[idx]:
                            option_documents.append(document['content'])
                        contexts.append(' '.join(option_documents))

                    question_inputs = self.retriever.tokenizer(query_options, contexts,
                                                               add_special_tokens=True,
                                                               max_length=512,
                                                               padding='max_length',
                                                               truncation=True,
                                                               return_tensors="pt")
                    input_ids.append(question_inputs['input_ids'])
                    token_type_ids.append(question_inputs['token_type_ids'])
                    attention_masks.append(question_inputs['attention_mask'])

                tensor_input_ids = torch.stack(input_ids, dim=0)
                tensor_token_type_ids = torch.stack(token_type_ids, dim=0)
                tensor_attention_masks = torch.stack(attention_masks, dim=0)

                outputs = self.reader.model(input_ids=tensor_input_ids.to(device),
                                            attention_mask=tensor_token_type_ids.to(
                                                device),
                                            token_type_ids=tensor_attention_masks.to(
                                                device),
                                            labels=answers_indexes.to(device))
                # loss [0] and classification scores [1], potential TODO: check if the Base_BERT_reader returns same valuess
                loss = outputs[0]

                if self.num_gpus > 1:
                    loss = loss.mean()
                total_train_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.reader.model.parameters(), 1.0)
                optimizer.step()

                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.4f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.retriever.q_encoder.eval()
            self.reader.model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0

            # Evaluate data for one epoch
            for step, batch in enumerate(val_dataloader):
                if step % 25 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print(
                        f'Batch {step} of {len(val_dataloader)}. Elapsed: {elapsed}')
                questions = batch[0]
                answers_indexes = batch[2]
                options = batch[3]
                metamap_phrases = batch[4]

                input_ids = []
                token_type_ids = []
                attention_masks = []

                for q_idx in range(len(questions)):
                    query = ' '.join(metamap_phrases[q_idx])
                    query_options = [query + ' ' + x[q_idx] for x in options]
                    retrieved_documents = [
                        self.retriever.retrieve_documents(x) for x in query_options]

                    contexts = []
                    for idx in range(len(retrieved_documents)):
                        option_documents = []
                        for document in retrieved_documents[idx]:
                            option_documents.append(document['content'])
                        contexts.append(' '.join(option_documents))

                    question_inputs = self.retriever.tokenizer(query_options, contexts,
                                                               add_special_tokens=True,
                                                               max_length=512,
                                                               padding='max_length',
                                                               truncation=True,
                                                               return_tensors="pt")
                    input_ids.append(question_inputs['input_ids'])
                    token_type_ids.append(question_inputs['token_type_ids'])
                    attention_masks.append(question_inputs['attention_mask'])

                tensor_input_ids = torch.stack(input_ids, dim=0)
                tensor_token_type_ids = torch.stack(token_type_ids, dim=0)
                tensor_attention_masks = torch.stack(attention_masks, dim=0)
                with torch.no_grad():
                    outputs = self.reader.model(input_ids=tensor_input_ids.to(device),
                                                attention_mask=tensor_token_type_ids.to(
                                                    device),
                                                token_type_ids=tensor_attention_masks.to(
                                                    device),
                                                labels=answers_indexes.to(device))
                loss, classification_scores = outputs[:2]
                if self.num_gpus > 1:
                    loss = loss.mean()
                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                if device.type == 'cpu':
                    classification_scores = classification_scores.numpy()
                    answers_indexes = answers_indexes.numpy()
                else:
                    classification_scores = classification_scores.detach().cpu().numpy()
                    answers_indexes = answers_indexes.to('cpu').numpy()
                total_eval_accuracy += self.calculate_accuracy(
                    classification_scores, answers_indexes)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
            print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

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
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")

        total_training_time = self.format_time(time.time()-total_t0)
        training_info['total_training_time'] = total_training_time
        print(f"Total training took {training_time} (h:mm:ss)")

        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
        # saving training stats
        training_stats_file = f"src/trainer/results/{dt_string}__REALM_like+base_BERT__training_stats.json"
        with open(training_stats_file, 'w') as results_file:
            json.dump(training_info, results_file)
        print(f"Results saved in {training_stats_file}")
        # saving the retriever's q_encoder weights
        retriever_file_name = f"src/trainer/results/{dt_string}__REALM_like+base_BERT__retriever.pth"
        torch.save(self.retriever.q_encoder.state_dict(), retriever_file_name)
        print(f"Q_encoder weights saved in {retriever_file_name}")
        # saving the reader weights
        reader_file_name = f"src/trainer/results/{dt_string}__REALM_like+base_BERT__reader.pth"
        torch.save(self.reader.model.state_dict(), reader_file_name)
        print(f"Reader weights saved in {retriever_file_name}")

        print("***** Training completed *****")
