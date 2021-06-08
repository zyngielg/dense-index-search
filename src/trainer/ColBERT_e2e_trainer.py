import datetime
import json
import numpy as np
import random
import time
import torch

from data.data_loader import create_questions_data_loader
from data.medqa_questions import MedQAQuestions
from retriever.colbert.ColBERT_retriever import ColBERT_retriever
from trainer.trainer import Trainer
from transformers import get_linear_schedule_with_warmup
from utils.general_utils import remove_duplicates_preserve_order

class ColBERT_e2e_trainer(Trainer):
    def __init__(self, questions: MedQAQuestions, retriever: ColBERT_retriever, num_epochs: int, batch_size: int, lr: float) -> None:
        super().__init__(questions, retriever, None, num_epochs, batch_size, lr)
        self.batch_size = 4

    def pepare_data_loader(self):
        # print("******** Creating train dataloader ********")
        # train_dataloader = create_questions_data_loader(
        #     questions=self.questions_train, tokenizer=self.retriever.tokenizer, batch_size=self.batch_size)
        # print("******** Train dataloader created  ********")

        print("******** Creating val dataloader ********")
        val_dataloader = create_questions_data_loader(
            questions=self.questions_val, tokenizer=self.retriever.tokenizer, batch_size=self.batch_size)
        print("******** Val dataloader created  ********")

        # return train_dataloader, val_dataloader
        return  val_dataloader, val_dataloader


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
            "retriever": self.retriever.get_info(),
            "reader": "None - ColBERT e2e",
            "total_training_time": None,
            "training_stats": []
        }

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.retriever.colbert.parameters(), lr=self.lr)

        total_steps = self.num_epochs * self.batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        train_dataloader, val_dataloader = self.pepare_data_loader()

        for epoch in range(self.num_epochs):
            print(f'======== Epoch {epoch + 1} / {self.num_epochs} ========')
            t0 = time.time()
            total_train_loss = 0
            self.retriever.colbert.train()
            for step, batch in enumerate(train_dataloader):
                if step % 10 == 0 and not step == 0:
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
                    metamap_phrases[q_idx] = remove_duplicates_preserve_order(metamap_phrases[q_idx])
                    query = ' '.join(metamap_phrases[q_idx])
                    query_options = [x + ' ' + query for x in options[q_idx]]

                    retrieved_documents, scores = self.retriever.retrieve_documents(query_options)
                    ### END OF DOCUMENT RETRIEVAL ###


                    ### BEGINNING OF RECALCULATING RETRIEVED DOCUMENTS SCORES
                    num_docs_retrieved = self.retriever.num_documents_reader
                    q_ids, q_mask = self.retriever.tokenizer.tensorize_queries(query_options)

                    retrieved_documents_reshaped = []
                    
                    for i in range(len(retrieved_documents[0])):
                        for j in range(len(retrieved_documents)):
                            retrieved_documents_reshaped.append(retrieved_documents[j][i])

                    # test_retrieved_documents = [item for sublist in retrieved_documents for item in sublist]
                    d_ids, d_mask = self.retriever.tokenizer.tensorize_documents(retrieved_documents_reshaped)
                    d_ids, d_mask = d_ids.view(num_docs_retrieved, len(query_options), -1), d_mask.view(num_docs_retrieved, len(query_options), -1)
                    
                    d_ids_stacked = [d_ids[i] for i in range(num_docs_retrieved)]
                    d_mask_stacked = [d_mask[i] for i in range(num_docs_retrieved)]

                    q_ids_stacked = [q_ids for i in range(num_docs_retrieved)]
                    q_mask_stacked = [q_mask for i in range(num_docs_retrieved)]
                                
                    Q = (torch.cat(q_ids_stacked), torch.cat(q_mask_stacked))
                    D = (torch.cat(d_ids_stacked), torch.cat(d_mask_stacked))

                    test = self.retriever.colbert(Q, D).view(num_docs_retrieved, -1).permute(1, 0)
                    test_mean = torch.mean(test, dim=1).unsqueeze(0)
                    results.append(test_mean)


                results = torch.cat(results)
                loss = criterion(results, answers_indexes.to(device))                
                if self.num_gpus > 1:
                    loss = loss.mean()
                total_train_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.retriever.colbert.parameters(), 1.0)

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
            self.retriever.colbert.eval()
            self.reader.model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0

            # Evaluate data for one epoch
            for step, batch in enumerate(val_dataloader):
                if step % 25 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print(
                        f'Batch {step} of {len(train_dataloader)}. Elapsed: {elapsed}')
                questions = batch[0]
                answers_indexes = batch[2]
                options = batch[3]
                metamap_phrases = batch[4]

                input_ids = []
                token_type_ids = []
                attention_masks = []

                for q_idx in range(len(questions)):
                    query = ' '.join(metamap_phrases[q_idx])
                    # or
                    # query = questions[q_idx]
                    query_options = [x + ' ' + query for x in options[q_idx]]
                    # retrieved_documents = [
                    #     self.retriever.retrieve_documents(x) for x in query_options]
                    retrieved_documents, scores = self.retriever.retrieve_documents(query_options)

                    contexts = []
                    for idx in range(len(retrieved_documents)):
                        option_documents = []
                        for document in retrieved_documents[idx]:
                            option_documents.append(document['content'])
                        contexts.append(' '.join(option_documents))

                    question_inputs = self.retriever.tokenizer(
                        query_options, contexts, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors="pt")
                    input_ids.append(question_inputs['input_ids'])
                    token_type_ids.append(question_inputs['token_type_ids'])
                    attention_masks.append(question_inputs['attention_mask'])

                tensor_input_ids = torch.stack(input_ids, dim=0)
                tensor_token_type_ids = torch.stack(token_type_ids, dim=0)
                tensor_attention_masks = torch.stack(attention_masks, dim=0)
                with torch.no_grad():
                    output = self.reader.model(
                        input_ids=tensor_input_ids.to(device), attention_mask=tensor_token_type_ids.to(device), token_type_ids=tensor_attention_masks.to(device))
                loss = criterion(output, answers_indexes.to(device))
                if self.num_gpus > 1:
                    loss = loss.mean()
                # Accumulate the validation loss.
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

                break

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

            print(f"Num of issues: {self.retriever.score_calc.issue_counter}")

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
