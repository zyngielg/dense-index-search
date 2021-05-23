import datetime
import json
import numpy as np
import random
import time
import torch

from data.data_loader import create_medqa_data_loader
from data.medqa_questions import MedQAQuestions
from reader.reader import Reader
from retriever.retriever import Retriever
from trainer.trainer import Trainer
from transformers import get_linear_schedule_with_warmup

# TODO: currently after 3 epochs the loss and the accuracy remain the same
# check whether the input to BERT should be reformatted compared to the current one


class IrEsBaseBertTrainer(Trainer):
    def __init__(self, questions: MedQAQuestions, retriever: Retriever, reader: Reader, num_epochs: int, batch_size: int, lr: float) -> None:
        super().__init__(questions, retriever, reader, num_epochs, batch_size, lr)

    def pepare_data_loader(self):
        print("******** Creating train dataloader ********")
        train_input_queries, train_input_answers, train_input_answers_idx = self.retriever.create_tokenized_input(
            questions=self.questions_train, tokenizer=self.reader.tokenizer, train_set=True)

        train_dataloader = create_medqa_data_loader(input_queries=train_input_queries, input_answers=train_input_answers,
                                                    input_answers_idx=train_input_answers_idx, batch_size=self.batch_size)
        print("******** Train dataloader created  ********")

        print("******** Creating val dataloader ********")

        val_input_queries, val_input_answers, val_input_answers_idx = self.retriever.create_tokenized_input(
            questions=self.questions_val, tokenizer=self.reader.tokenizer, train_set=False)
        val_dataloader = create_medqa_data_loader(input_queries=val_input_queries, input_answers=val_input_answers,
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
        total_t0 = time.time()

        training_info = {
            "retriever": self.retriever.get_info(),
            "reader": self.reader.get_info(),
            "total_training_time": None,
            "training_stats": []
        }

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.reader.model.parameters(), lr=self.lr)

        total_steps = self.num_epochs * self.batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        train_dataloader, val_dataloader = self.pepare_data_loader()

        for epoch in range(self.num_epochs):
            print(f'======== Epoch {epoch + 1} / {self.num_epochs} ========')
            t0 = time.time()
            total_train_loss = 0
            self.reader.model.train()
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

                output = self.reader.model(input_ids=input_ids.to(device),
                                           attention_mask=input_attention_mask.to(
                                               device),
                                           token_type_ids=input_token_type_ids.to(device))

                loss = criterion(output, answers_indexes.to(device))
                if self.num_gpus > 1:
                    loss = loss.mean()
                total_train_loss += loss.item()
                loss.backward()
                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(
                    self.reader.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
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
        training_stats_file = f"src/trainer/results/{dt_string}__training_stats.json"
        with open(training_stats_file, 'w') as results_file:
            json.dump(training_info, results_file)
        # saving the reader weights
        reader_file_name = f"src/trainer/results/{dt_string}__IRES+base_BERT__reader.pth"
        torch.save(self.reader.model.state_dict(), reader_file_name)
        print(f"Reader weights saved in {reader_file_name}")
        print("***** Training completed *****")
