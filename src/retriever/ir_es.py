from retriever.retriever import Retriever
from elasticsearch import Elasticsearch
from tqdm import tqdm
from utils.es_utils import Indexes
import utils.es_utils as es_utils
import numpy as np

import utils.pickle_utils as pickle_utils


class IR_ES(Retriever):
    stemmed = True

    index_name = Indexes.MedQA_chunks_50.value
    num_of_documents_to_retrieve = 20
    host = ['http://localhost']
    port = '9200'

    retrieved_documents_train_path = "data/es-retrieved-documents/es_retrieved_documents_train_chunks_50_questions_unprocessed.pickle"
    retrieved_documents_val_path = "data/es-retrieved-documents/es_retrieved_documents_val_chunks_50_questions_unprocessed.pickle"

    def __init__(self, from_es_session=False):
        self.from_es_session = from_es_session

        if self.from_es_session is True:
            self.es = Elasticsearch(hosts=self.host, port=self.port)
            if not self.es.ping():
                raise ValueError(
                    "Connection failed. Make sure that elasticsearch instance is running")
            self.setup_elasticsearch()
        else:
            self.train_retrieved_documents = pickle_utils.load_pickle(
                self.retrieved_documents_train_path)
            self.val_retrieved_documents = pickle_utils.load_pickle(
                self.retrieved_documents_val_path)

    def setup_elasticsearch(self):
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

    def create_tokenized_input(self, questions, tokenizer, train_set):
        def letter_answer_to_index(answer):
            return ord(answer) - 65

        input_queries = []
        input_answers = []
        input_answers_idx = []

        for question_id, question_data in tqdm(questions.items()):
            question = question_data['question']
            metamap_phrases = question_data['metamap_phrases']
            queries = []
            for option in question_data['options'].values():
                qa_retrieval = ' '.join(metamap_phrases) + ' ' + option
                qa_inference = f"{question} {option}"
                
                _, retrieved_documents = self.retrieve_documents(
                    query=qa_retrieval, question_id=question_id, option=option.strip().lower(), train=train_set)

                context = ' '.join(retrieved_documents)
                query = tokenizer(context, qa_inference, add_special_tokens=True,
                                  max_length=512, padding='max_length', truncation=True, return_tensors="pt")
                query_input_ids = query["input_ids"].flatten()
                # decoded = tokenizer.decode(query_input_ids)
                query_token_type_ids = query["token_type_ids"].flatten()
                query_attention_mask = query["attention_mask"].flatten()

            queries.append({
                "input_ids": query_input_ids,
                "token_type_ids": query_token_type_ids,
                "attention_mask": query_attention_mask
            })
            input_queries.append(queries)
            input_answers.append(question_data["answer"])

            input_answers_idx.append(
                letter_answer_to_index(question_data['answer_idx']))
        return input_queries, input_answers, input_answers_idx

    def retrieve_documents(self, query=None, question_id=None, option=None, train=False):
        retrieved_documents = None
        retrieved_documents_content = None

        if self.from_es_session is True:
            retrieved_documents = es_utils.search_documents(es=self.es,
                                                            query_input=query,
                                                            n=self.num_of_documents_to_retrieve,
                                                            index_name=self.index_name,
                                                            stemmed=self.stemmed)
        else:
            if train:
                retrieved_documents = self.train_retrieved_documents[
                    question_id]['retrieved_documents'][option]
            else:
                retrieved_documents = self.val_retrieved_documents[
                    question_id]['retrieved_documents'][option]
        retrieved_documents_content = [
            x['evidence']['content'] for x in retrieved_documents]
        return retrieved_documents, retrieved_documents_content

    def calculate_score(self, retrieved_documents):
        return np.sum([doc['score'] for doc in retrieved_documents])

    def run_ir_es_e2e(self, questions):
        correct_answer = 0
        incorrect_answer = 0

        for question_id, question_data in tqdm(questions.items()):
            question = question_data['question']
            answer = question_data['answer']

            final_answer = None
            final_score = 0

            for option, option_answer in question_data['options'].items():
                # query = ' '.join(question_data['metamap_phrases']) + " " + option_answer
                query = question + " " + option_answer
                top_documents, _ = self.retrieve_documents(query)
                if top_documents != []:
                    score = self.calculate_score(top_documents)
                    if final_score < score:
                        final_answer = option_answer
                        final_score = score
            correct = False
            if final_answer == answer:
                correct_answer += 1
                correct = True
            else:
                incorrect_answer += 1

        print(
            f'Accuracy: {100 * correct_answer / (correct_answer + incorrect_answer)}%')
        print(f'\tCorrect answers: {correct_answer}')
        print(f'\tInorrect answers: {incorrect_answer}')
