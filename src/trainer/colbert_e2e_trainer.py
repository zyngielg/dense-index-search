import datetime
import json
import numpy as np
import random
import time
import torch

from data.data_loader import create_questions_data_loader
from data.medqa_questions import MedQAQuestions
from retriever.colbert.colbert_retriever import ColBERTRetriever
from trainer.trainer import Trainer
from transformers import get_linear_schedule_with_warmup
from utils.general_utils import remove_duplicates_preserve_order

class ColBERTe2eTrainer(Trainer):
    def __init__(self, questions: MedQAQuestions, retriever: ColBERTRetriever, num_epochs: int, batch_size: int, lr: float) -> None:
        super().__init__(questions, retriever, None, num_epochs, batch_size, lr)
        self.batch_size = 32
        self.num_train_questions = len(self.questions_train)
        self.num_val_questions = len(self.questions_val)

    def pepare_data_loader(self):
        print("******** Creating train dataloader ********")
        train_dataloader = create_questions_data_loader(
            questions=self.questions_train, batch_size=self.batch_size, num_questions=self.num_train_questions)
        print("******** Train dataloader created  ********")

        print("******** Creating val dataloader ********")
        val_dataloader = create_questions_data_loader(
            questions=self.questions_val, batch_size=self.batch_size, num_questions=self.num_val_questions)
        print("******** Val dataloader created  ********")

        return train_dataloader, val_dataloader
        # return  val_dataloader, val_dataloader


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

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.retriever.colbert.parameters(), lr=self.lr)

        total_steps = self.num_epochs * self.batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=500,
                                                    num_training_steps=total_steps)

        train_dataloader, val_dataloader = self.pepare_data_loader()
        for epoch in range(self.num_epochs):
            print(f'======== Epoch {epoch + 1} / {self.num_epochs} ========')
            t0 = time.time()
            total_train_loss = 0
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
                    metamap_phrases[q_idx] = remove_duplicates_preserve_order(metamap_phrases[q_idx])
                    query = ' '.join(metamap_phrases[q_idx])
                    query_options = [query + f' {self.retriever.tokenizer.option_token} ' + x for x in options[q_idx]]
                    # query_options = ['[unused5] ' + x + ' [unused6] ' + query for x in options[q_idx]]
                    retrieved_documents, scores = self.retriever.retrieve_documents(query_options)
                    scores_mean = torch.mean((scores), dim=1)
                    results.append(scores_mean)


                results = torch.stack(results)
                loss = criterion(results, answers_indexes.to("cuda:2"))                
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
            self.retriever.colbert.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0

            # Evaluate data for one epoch
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
                        ### BEGINNING OF DOCUMENT RETRIEVAL ###
                        metamap_phrases[q_idx] = remove_duplicates_preserve_order(metamap_phrases[q_idx])
                        query = ' '.join(metamap_phrases[q_idx])
                        query_options = [query + f' {self.retriever.tokenizer.option_token} ' + x for x in options[q_idx]]
                        # query_options = ['[unused5] ' + x + ' [unused6] ' + query for x in options[q_idx]]

                        retrieved_documents, scores = self.retriever.retrieve_documents(query_options)
                        ### END OF DOCUMENT RETRIEVAL ###
                        scores_mean = torch.mean((scores), dim=1)
                        results.append(scores_mean)
                        # ### BEGINNING OF RECALCULATING RETRIEVED DOCUMENTS SCORES
                        # num_docs_retrieved = self.retriever.num_documents_reader
                        # q_ids, q_mask = self.retriever.tokenizer.tensorize_queries(query_options)

                        # retrieved_documents_reshaped = []
                        
                        # for i in range(len(retrieved_documents[0])):
                        #     for j in range(len(retrieved_documents)):
                        #         retrieved_documents_reshaped.append(retrieved_documents[j][i])

                        # # test_retrieved_documents = [item for sublist in retrieved_documents for item in sublist]
                        # d_ids, d_mask = self.retriever.tokenizer.tensorize_documents(retrieved_documents_reshaped)
                        # d_ids, d_mask = d_ids.view(num_docs_retrieved, len(query_options), -1), d_mask.view(num_docs_retrieved, len(query_options), -1)
                        
                        # d_ids_stacked = [d_ids[i] for i in range(num_docs_retrieved)]
                        # d_mask_stacked = [d_mask[i] for i in range(num_docs_retrieved)]

                        # q_ids_stacked = [q_ids for i in range(num_docs_retrieved)]
                        # q_mask_stacked = [q_mask for i in range(num_docs_retrieved)]
                                    
                        # Q = (torch.cat(q_ids_stacked), torch.cat(q_mask_stacked))
                        # D = (torch.cat(d_ids_stacked), torch.cat(d_mask_stacked))

                        # test = self.retriever.colbert(Q, D).view(num_docs_retrieved, -1).permute(1, 0)
                        # test_mean = torch.mean(test, dim=1).unsqueeze(0)
                        # results.append(test_mean)


                results = torch.stack(results)
                loss = criterion(results, answers_indexes.to("cuda:2"))   
                if self.num_gpus > 1:
                    loss = loss.mean()
                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                if device.type == 'cpu':
                    output = results.numpy()
                    answers_indexes = answers_indexes.numpy()
                else:
                    output = results.detach().cpu().numpy()
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
                    'Training Time': training_time,
                    'Training Loss': avg_train_loss,
                    'Validation Time': validation_time,
                    'Validation Loss': avg_val_loss,
                    'Validation Accuracy.': avg_val_accuracy
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
        training_stats_file = f"src/results/colbert-based/{dt_string}__ColBERT_e2e_stats.json"
        with open(training_stats_file, 'w') as results_file:
            json.dump(training_info, results_file)
        print(f"Results saved in {training_stats_file}")
      
        colbert_file_name = f"src/results/colbert-based/{dt_string}__ColBERT_e2e_retriever.pth"
        torch.save(self.retriever.colbert.state_dict(), colbert_file_name)
        print(f"Reader weights saved in {colbert_file_name}")

        print("***** Training completed *****")
