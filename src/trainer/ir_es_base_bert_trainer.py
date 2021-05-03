from reader.base_bert_reader import Base_BERT_Reader
from retriever.ir_es import IR_ES
from data.medqa_questions import MedQAQuestions
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import random
import time
import torch
import time
import datetime
from data.data_loader import create_data_loader


class IrEsBaseBertTrainer():
    def __init__(self, questions: MedQAQuestions, retriever: IR_ES, reader: Base_BERT_Reader):
        self.retriever = retriever
        self.reader = reader
        self.questions_train = questions.questions_train
        self.questions_val = questions.questions_val
        self.num_epochs = 3
        self.batch_size = 32
        self.lr = 5e-5

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
        predictions = np.argmax(predictions_distribution, axis=1)
        return np.sum(predictions == correct_answers) / len(correct_answers)

    def pepare_data_loader(self):
        print("******** Creating train dataloader ********")
        train_input_queries, train_input_answers, train_input_answers_idx = self.retriever.create_tokenized_input(
            questions=self.questions_train, tokenizer=self.reader.tokenizer)
        
        train_dataloader = create_data_loader(input_queries=train_input_queries, input_answers=train_input_answers,
                                              input_answers_idx=train_input_answers_idx, batch_size=self.batch_size)
        print("******** Train dataloader created  ********")
        
        print("******** Creating val dataloader ********")

        val_input_queries, val_input_answers, val_input_answers_idx = self.retriever.create_tokenized_input(
            questions=self.questions_val, tokenizer=self.reader.tokenizer)
        val_dataloader = create_data_loader(input_queries=val_input_queries, input_answers=val_input_answers,
                                            input_answers_idx=val_input_answers_idx, batch_size=self.batch_size)
        print("******** Val dataloader created  ********")
        return train_dataloader, val_dataloader

    def train(self):
        print("***** Running training *****")
        device = self.reader.device
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        total_t0 = time.time()

        training_stats = []

        #  We fine-tune for 3 epochs with a learning rate of 5e-5 and a batch size of 32
        lr = 5e-5

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
                if step % 10 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print(
                        f'Batch {step} of {len(train_dataloader)}. Elapsed: {elapsed}')

                self.reader.model.zero_grad()  # no difference if model or optimizer.zero_grad

                questions_queries_collection = batch[0]
                answers = batch[1]
                answers_indexes = batch[2]
                queries_outputs = []
                for question_queries in questions_queries_collection:
                    input_ids = question_queries["input_ids"].to(device)
                    input_token_type_ids = question_queries["token_type_ids"].to(
                        device)
                    input_attention_mask = question_queries["attention_mask"].to(
                        device)

                    # the forward pass, since this is only needed for backprop (training).
                    # Tell pytorch not to bother with constructing the compute graph during
                    output = self.reader.model(
                        input_ids=input_ids, attention_mask=input_attention_mask, token_type_ids=input_token_type_ids)
                    queries_outputs.append(output)
                # each row represents values for the same question, each column represents an output for an answer option
                queries_outputs = torch.stack(queries_outputs).reshape(
                    [5, len(answers)]).transpose(0, 1)
                # choosing the indexes of the answers with the highest post-softmax value
                output = self.reader.model.softmax(queries_outputs)

                loss = criterion(output, answers_indexes.to(device))
                total_train_loss += loss
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
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
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
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for step, batch in enumerate(val_dataloader):
                if step > 10:
                    continue
                questions_queries_collection = batch[0]
                answers = batch[1]
                answers_indexes = batch[2]

                queries_outputs = []
                for question_queries in questions_queries_collection:
                    input_ids = question_queries["input_ids"].to(device)
                    input_token_type_ids = question_queries["token_type_ids"].to(
                        device)
                    input_attention_mask = question_queries["attention_mask"].to(
                        device)

                    # Tell pytorch not to bother with constructing the compute graph during
                    # the forward pass, since this is only needed for backprop (training).
                    output = self.reader.model(
                        input_ids=input_ids, attention_mask=input_attention_mask, token_type_ids=input_token_type_ids)
                    queries_outputs.append(output)

                queries_outputs = torch.stack(queries_outputs).reshape(
                    [5, len(answers)]).transpose(0, 1)
                output = self.reader.model.softmax(queries_outputs)
                loss = criterion(output, answers_indexes.to(device))

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                if device == 'cpu':
                    output = output.numpy()
                    answers_indexes = answers_indexes.numpy()
                else:
                    output = output.detach().cpu().numpy()
                    answers_indexes = answers_indexes.to('cpu').numpy()
                total_eval_accuracy += self.calculate_accuracy(
                    output, answers_indexes)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(val_dataloader)

            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
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

        print("Total training took {:} (h:mm:ss)".format(
            self.format_time(time.time()-total_t0)))
        
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
        model_name = f"src/trainer/results/{dt_string}__reader:IRES__retriever:BERT_linear.pth"
        torch.save(self.reader.model.state_dict(), model_name)
        print(f"Model weights saved in {model_name}")
        print("***** Training completed *****")
