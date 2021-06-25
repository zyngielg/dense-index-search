import numpy as np
import torch
import utils.es_utils as es_utils
import utils.pickle_utils as pickle_utils
import json
import datetime
from elasticsearch import Elasticsearch
from retriever.retriever import Retriever
from tqdm import tqdm


class IR_ES(Retriever):
    stemmed = True

    index_name = es_utils.Indexes.MedQA_chunks_100.value
    num_of_documents_to_retrieve = 10
    host = ['http://localhost']
    port = '9200'

    retrieved_documents_train_path = "data/es-retrieved-documents/final_es_retrieved_documents_train_chunks_100_unprocessed.pickle"
    retrieved_documents_val_path = "data/es-retrieved-documents/final_es_retrieved_documents_val_chunks_100_unprocessed.pickle"

    def __init__(self, from_es_session=False):
        self.from_es_session = from_es_session

    def get_info(self):
        info = {}
        info['from es session'] = self.from_es_session
        if not self.from_es_session:
            info['train searches loaded from'] = self.retrieved_documents_train_path
            info['val searches loaded from'] = self.retrieved_documents_val_path
        else:
            info['index name'] = self.index_name
        info['num of docs retrieved'] = self.num_of_documents_to_retrieve

        return info

    def prepare_retriever(self, corpus=None, create_encodings=None, create_index=None):
        if self.from_es_session:
            self.es = Elasticsearch(hosts=self.host, port=self.port)
            if not self.es.ping():
                raise ValueError(
                    "Connection failed. Make sure that elasticsearch instance is running")
            self.__setup_elasticsearch()
        else:
            self.train_retrieved_documents = pickle_utils.load_pickle(
                self.retrieved_documents_train_path)
            self.val_retrieved_documents = pickle_utils.load_pickle(
                self.retrieved_documents_val_path)

    def __setup_elasticsearch(self):
        if not es_utils.check_if_index_exists(es=self.es,
                                              index_name=self.index_name):
            create_index_body = """{
                "settings": {
                    "index": {
                        "similarity": {
                            "default": {
                                "type": "BM25"
                            }
                        }
                    }
                }
            }"""
            es_utils.create_index(es=self.es,
                                  index_name=self.index_name,
                                  index_body=create_index_body)

    def create_tokenized_input(self, questions, tokenizer, docs_flag, num_questions, medqa=True):
        question_token_id = tokenizer.convert_tokens_to_ids('[unused5]')
        answer_token_id = tokenizer.convert_tokens_to_ids('[unused6]')

        def letter_answer_to_index(answer):
            return ord(answer) - 65

        input_queries = []
        input_answers = []
        input_answers_idx = []

        for question_id, question_data in tqdm(questions.items()):
            if int(question_id[1:]) == num_questions:
                break
            metamap_phrases = question_data['metamap_phrases']
            question_raw = question_data["question"]

            input_ids = []
            token_type_ids = []
            attention_masks = []

            for option in question_data['options'].values():
                medqa_string = ' '.join(metamap_phrases)
                query = f"{option} {medqa_string}"

                _, retrieved_documents = self.retrieve_documents(
                    query=query,
                    question_id=question_id,
                    option=option,
                    retrieved_docs_flag=docs_flag)
                if medqa:
                    query = f"{option} . {medqa_string}"
                else:
                    query = f"{option} . {question_raw}"
                tokenized_option_len = len(tokenizer(option, add_special_tokens=False)['input_ids'])

                context = ' '.join(retrieved_documents)
                query_tokenized = tokenizer(query, context, add_special_tokens=True,
                                            max_length=512, padding='max_length', truncation='longest_first', return_tensors="pt")
                query_tokenized['input_ids'][:, 1 + tokenized_option_len] = answer_token_id

                input_ids.append(query_tokenized["input_ids"].flatten())
                token_type_ids.append(
                    query_tokenized["token_type_ids"].flatten())
                attention_masks.append(
                    query_tokenized["attention_mask"].flatten())

            tensor_input_ids = torch.stack(input_ids, dim=0)
            tensor_token_type_ids = torch.stack(token_type_ids, dim=0)
            tensor_attention_masks = torch.stack(attention_masks, dim=0)

            input_queries.append({
                "input_ids": tensor_input_ids,
                "token_type_ids": tensor_token_type_ids,
                "attention_mask": tensor_attention_masks
            })
            input_answers.append(question_data["answer"])

            input_answers_idx.append(
                letter_answer_to_index(question_data['answer_idx']))

        return input_queries, input_answers, input_answers_idx

    def retrieve_documents(self, query=None, question_id=None, option=None, retrieved_docs_flag=0, question_raw=None):
        retrieved_documents = None
        retrieved_documents_content = None

        if self.from_es_session is True:
            retrieved_documents = es_utils.search_documents(es=self.es,
                                                            query_input=query,
                                                            n=self.num_of_documents_to_retrieve,
                                                            index_name=self.index_name,
                                                            stemmed=self.stemmed)
        else:
            if retrieved_docs_flag == 0:
                retrieved_docs_set = self.train_retrieved_documents
            elif retrieved_docs_flag == 1:
                retrieved_docs_set = self.val_retrieved_documents
            # else:
                # retrieved_docs_set = self.test_retrieved_documents
            if not question_raw:
                retrieved_documents = retrieved_docs_set[question_id]['retrieved_documents'][option]
            else:
                for key, val in retrieved_docs_set.items():
                    if val['question'] == question_raw:
                        question_id = key
                        break
                x = retrieved_docs_set[key]
                retrieved_documents = retrieved_docs_set[key]['retrieved_documents'][option]

        retrieved_documents_content = [
            x['evidence']['content'] for x in retrieved_documents]
        return retrieved_documents, retrieved_documents_content

    def __calculate_score(self, retrieved_documents):
        return np.sum([doc['score'] for doc in retrieved_documents])

    def save_results(self, train_results, val_results):
        results = {
            "info": self.get_info(),
            "results": [train_results, val_results]
        }
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
        results_file = f"src/results/ir-es-based/{dt_string}__IR-ES__e2e.json"
        with open(results_file, 'w') as results_file:
            json.dump(results, results_file)

    def run_ir_es_e2e(self, questions, doc_flag=0):
        correct_answer = 0
        incorrect_answer = 0

        for q_id, question_data in tqdm(questions.items()):
            question_raw = question_data['question']
            answer = question_data['answer']

            final_answer = None
            final_score = 0

            for option_answer in question_data['options'].values():
                metamap_string = ' '.join(question_data['metamap_phrases'])
                query = f"{option_answer} {metamap_string}"
                top_documents, _ = self.retrieve_documents(
                    query=query.lower(),
                    question_id=q_id,
                    retrieved_docs_flag=doc_flag,
                    option=option_answer)
                if top_documents != []:
                    score = self.__calculate_score(top_documents)
                    if final_score < score:
                        final_answer = option_answer
                        final_score = score

            if final_answer == answer:
                correct_answer += 1
            else:
                incorrect_answer += 1

        accuracy = 100 * correct_answer / (correct_answer + incorrect_answer)
        results = {
            "Accuracy": accuracy,
            "# correct": correct_answer,
            "# incorrect": incorrect_answer
        }
        print(f'Accuracy: {accuracy}%')
        print(f'\tCorrect answers: {correct_answer}')
        print(f'\tInorrect answers: {incorrect_answer}')

        return results
