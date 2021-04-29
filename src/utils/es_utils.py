import math

from elasticsearch import Elasticsearch
from enum import Enum
from nltk import FreqDist, ngrams, word_tokenize
from numpy import mean


class Indexes(Enum):
    MedQA_chunks_100 = "medqa-chunks-100"
    MedQA_chunks_50 = "medqa-chunks-50"


def get_list_of_indexeses(es: Elasticsearch):
    return list(es.indices.get_alias("*").keys())


def check_if_index_exists(es: Elasticsearch, index_name):
    return es.indices.exists(index=index_name)


def create_index(es: Elasticsearch, index_name, index_body):
    es.indices.create(index=index_name, body=index_body)


def search_documents(es: Elasticsearch, query_input, n, index_name, stemmed=False):
    match_option = "content" if stemmed is False else "content_stemmed"
    body = {
        "query": {
            "match": {
                match_option: query_input
            }
        },
        "from": 0,
        "size": n
    }    

    res = es.search(
        index=index_name,
        body=body
    )

    number_of_hits = len(res['hits']['hits'])

    results = []
    for i in range(number_of_hits):
        score = res['hits']['hits'][i]['_score']
        paragraph = res['hits']['hits'][i]['_source']
        result = {
            "score": score,
            "evidence": paragraph
        }
        results.append(result)

    return results


def idf(q_i, documents):
    nominator = len(documents)
    denominator = 1
    for document_tokens in documents:
        if q_i in document_tokens:
            denominator += 1

    return math.log(nominator/denominator)


def f(q_i, content):
    content_freq_dist = FreqDist(word_tokenize(content))
    q_i_frequency = content_freq_dist.get(q_i)
    if q_i_frequency is None:
        return 0
    else:
        return q_i_frequency


def bm_25(q_i, Q, documents, avg_doc_len, k_d, b_d):
    doc_len = sum([len(word_tokenize(x)) for x in documents])
    combined_documents = ' '.join(documents)

    nominator = idf(q_i, documents) * f(q_i, combined_documents) * (k_d + 1)
    denominator = f(q_i, Q) + k_d + (1 - b_d + b_d * doc_len/avg_doc_len)

    return nominator/denominator


def ir_es_custom_score(documents, query, avg_query_len, avg_doc_len):
    k_q = 0.4
    b_q = 0.7
    k_d = 0.9
    b_d = 0.35

    # query_unigrams = [x[0] for x in list(ngrams(word_tokenize(query), 1))]
    query_unigrams = word_tokenize(query)
    query_len = len(query_unigrams)

    combined_document = ' '.join(documents)
    combined_document_tokens = word_tokenize(combined_document)

    score = 0
    for q_i in query_unigrams:
        bm25 = bm_25(q_i=q_i, Q=query, documents=documents,
                     avg_doc_len=avg_doc_len, k_d=k_d, b_d=b_d)
        idf_q_i = idf(q_i=q_i, documents=documents)
        f_q_i_Q = f(q_i=q_i, content=query)

        nominator = bm25 * idf_q_i * f_q_i_Q * (k_q + 1)
        denominator = f_q_i_Q + k_q * (1 - b_q + b_q * query_len/avg_query_len)

        score += nominator / denominator

    return score
