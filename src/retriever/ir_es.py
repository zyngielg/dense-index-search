from retriever.retriever import Retriever
from elasticsearch import Elasticsearch
from tqdm import tqdm
import utils.es_utils as es_utils
import numpy as np


class IR_ES(Retriever):
    index_name = "medqa-unprocessed-chunks"
    num_of_documents_to_retrieve = 20
    host = ['http://localhost']
    port = '9200'

    def __init__(self):
        self.es = Elasticsearch(hosts=self.host, port=self.port)
        if not self.es.ping():
            raise ValueError("Connection failed. Make sure that elasticsearch instance is running")

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

    def retrieve_documents(self, query):
        retrieved_documents = es_utils.search_documents(es=self.es,
                                                        query_input=query,
                                                        n=self.num_of_documents_to_retrieve,
                                                        index_name=self.index_name)
        return retrieved_documents, [x['evidence']['content'] for x in retrieved_documents]

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
