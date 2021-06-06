import faiss
import torch
import numpy as np
import faiss
import json
import math
from data.medqa_corpus import MedQACorpus
from retriever.retriever import Retriever
from transformers import AutoTokenizer, AutoModel
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
from utils.pickle_utils import save_pickle, load_pickle
from models.ColBERT import ColBERT
from retriever.colbert.ColBERT_tokenizer import ColbertTokenizer


class ColBERT_like_retriever(Retriever):

    bert_name = "emilyalsentzer/Bio_ClinicalBERT"
    # change to specify the weights file
    q_encoder_weights_path = ""
    num_documents = 5
    layers_to_not_freeze = ['9', '10', '11', 'pooler']

    stemmer = SnowballStemmer(language='english')

    bert_weights_path = "data/colbert-clinical-biobert-cosine-200000.dnn"
    faiss_index_path = "data/index_colbert_l2_clinical_bert_chunks_100_non_processed.index"
    document_embeddings_path = "data/document_embeddings_colbert_cosine_200000_clinical_bert_chunks_100_non_processed.pickle"
    embbedding2doc_id_path = "data/embbedding2doc_id_path_colbert_cosine_200000_clinical_bert_chunks_100_non_processed.pickle"
    chunk_150_unstemmed_path = "data/chunks_150_non_processed.pickle"
    chunk_100_unstemmed_path = "data/chunks_100_non_processed.pickle"

    def __init__(self, load_weights=False, load_index=False) -> None:
        super().__init__()
        self.load_weights = load_weights
        self.load_index = load_index
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using {} device".format(self.device))

        # defining tokenizer and encoders
        self.colbert = ColBERT.from_pretrained(self.bert_name,
                                               query_maxlen=512,
                                               doc_maxlen=512,
                                               device=self.device,
                                               dim=128)
        self.tokenizer = ColbertTokenizer(bert_name=self.bert_name,
                                          query_maxlen=self.colbert.query_maxlen,
                                          doc_maxlen=self.colbert.query_maxlen)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.colbert = torch.nn.DataParallel(self.colbert)

        # loading weights
        if load_weights:
            ColBERT.load_checkpoint(self.bert_weights_path, self.colbert)

        # loading index
        if load_index:
            self.index = faiss.read_index(self.faiss_index_path)

        # freezing layers
        self.freeze_layers()
        # self.colbert.to(self.device)

        # info about used chunks
        self.used_chunks_size = 100

    def get_info(self):
        info = {}
        info['num documents retrieved'] = self.num_documents

        info['colbert'] = self.bert_name
        info['layers not to freeze'] = self.layers_to_not_freeze
        info['weights loaded'] = self.load_weights
        if self.load_weights:
            info['weights path'] = self.q_encoder_weights_path
        info['index loaded'] = self.load_index
        if self.load_index:
            info['index path'] = self.faiss_index_path
        info['chunk_size_used'] = self.used_chunks_size

        return info

    def retrieve_documents(self, queries: list):
        input_ids, mask = self.tokenizer.tensorize_queries(queries)
        Q = self.colbert.module.query(input_ids, mask)

        num_queries, embeddings_per_query, dim = Q.size()
        Q_faiss = Q.view(num_queries * embeddings_per_query,
                         dim).cpu().detach().numpy()

        scores, ids = self.index.search(Q_faiss, 5)

        ids = torch.from_numpy(ids)
        ids = ids.view(num_queries, embeddings_per_query * ids.size(1))

        all_doc_ids = self.embbedding2doc_id[ids].tolist()
        def uniq(l):
            return list(set(l))
        all_doc_ids = list(map(uniq, all_doc_ids))


        x = 2
        retrieved_documents = [self.corpus_chunks[i] for i in ids[0]]
        return retrieved_documents

    def freeze_layers(self):
        for name, param in self.colbert.named_parameters():
            if not any(x in name for x in self.layers_to_not_freeze):
                param.requires_grad = False
            # else:
            #     print(
            #         f"Layer {name} not frozen (status: {param.requires_grad})")

    def prepare_retriever(self, corpus: MedQACorpus = None, create_encodings=True, create_index=True):
        if self.used_chunks_size == 100:
            chunks_path = self.chunk_100_unstemmed_path
        elif self.used_chunks_size == 150:
            chunks_path = self.chunk_150_unstemmed_path

        if corpus is None:
            print(
                f"Loading the corpus chunks of size {self.used_chunks_size} from {chunks_path}")
            self.corpus_chunks = load_pickle(chunks_path)
        else:
            self.corpus_chunks = self.__create_corpus_chunks(
                corpus=corpus.corpus, chunk_length=self.used_chunks_size)
            save_pickle(self.corpus_chunks, chunks_path)

        dimension = self.colbert.module.dim  # NOT 768

        if create_encodings:
            print("******** 1a. Creating document embeddings ... ********")
            embeddings = []
            for idx, chunk in enumerate(tqdm(self.corpus_chunks)):
                content = chunk['content']

                ids, mask = self.tokenizer.tensorize_documents([content])
                # test = self.tokenizer.doc_tokenizer.decode(ids[0])
                with torch.no_grad():
                    embedding = self.colbert.module.doc(ids, mask)[0]
                embeddings.append(embedding)

            doc_embeddings_lengths = [d.size(0) for d in embeddings]
            embeddings = torch.cat(embeddings)
            assert dimension == embeddings.shape[-1]

            doc_embeddings = embeddings.float().numpy()
            print("********     ... embeddings created *********")

            print("******** 1b. Creating embedding2doc_id matrix ... ********")
            total_num_embeddings = sum(doc_embeddings_lengths)
            self.embbedding2doc_id = np.zeros(total_num_embeddings, dtype=int)
            
            offset = 0
            for doc_id, doc_length in enumerate(doc_embeddings_lengths):
                self.embbedding2doc_id[offset: offset + doc_length] = doc_id
                offset += doc_length
            print("******** 1b. ... matrix created ... ********")
            
            
            print("******** 1c. Saving document embeddings and embedding2doc_id to file ... ********")
            save_pickle(doc_embeddings, self.document_embeddings_path)
            save_pickle(self.embbedding2doc_id, self.embbedding2doc_id_path)
            print("********     ... embeddings and matrix saved *********")

        else:
            print("******** 1. Loading document embeddings and embedding2doc_id matrix... ********")
            doc_embeddings = load_pickle(self.document_embeddings_path)
            self.embbedding2doc_id = load_pickle(self.embbedding2doc_id_path)
            print("********    ... embeddings and matrix loaded ********")
        # if create_index:
        #     print("******** 2a. Creating and populating faiss index ...  *****")
        #     # num_embeddings = doc_embeddings.shape[0]
        #     # partitions = 1 << math.ceil(
        #     #     math.log2(8 * math.sqrt(num_embeddings)))

        #     # build a flat (CPU) index
        #     index = faiss.IndexFlatIP(dimension)
        #     if self.device.type != 'cpu':
        #         # index = faiss.index_cpu_to_all_gpus(index)
        #         res = faiss.StandardGpuResources()  # use a single GPU
        #         index = faiss.index_cpu_to_gpu(res, 1, index)
        #     # index.train(chunks_encodings)
        #     # index.add(chunks_encodings)
        #     # self.index = index

        #     index.add(doc_embeddings)

        #     self.index = index

        #     print("********      ... index created and populated ********")

        #     print("******** 2b. Saving the index ... ********")
        #     if self.device.type != 'cpu':
        #         index = faiss.index_gpu_to_cpu(index)
        #     faiss.write_index(index, self.faiss_index_path)
        #     print("********     ... index saved ********")
        # else:
        #     print("******** 2. Loading faiss index ...  ********")
        #     index = faiss.read_index(self.faiss_index_path)
        #     res = faiss.StandardGpuResources()  # use a single GPU
        #     self.index = faiss.index_cpu_to_gpu(res, 1, index)

            

        #     print("********    ... index loaded ********")

        print("******** 3. Creatin  ********")

        print("*** Finished Preparing the ColBERT retriever ***")

    def __preprocess_content(self, content, remove_stopwords, stemming, remove_punctuation):
        if not remove_stopwords and not stemming and not remove_punctuation:
            return content.lower().strip()
        # if remove_punctuation:
        #     content = content.translate(punctuation).replace(
        #         '“', '').replace('’', '')
        sentences = sent_tokenize(content.lower().strip())
        cleaned_sentences = []

        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            # if remove_stopwords:
            #     tokens = [x for x in tokens if x not in stop_words]
            if stemming:
                tokens = [self.stemmer.stem(x) for x in tokens]
            cleaned_sentences.append(' '.join(tokens))

        return ' '.join(cleaned_sentences)

    def __create_corpus_chunks(self, corpus, chunk_length):

        corpus_chunks = []
        for title, content in tqdm(corpus.items()):

            content_tokens = word_tokenize(content)

            counter = 0
            for i in range(0, len(content_tokens), chunk_length):
                chunk_name = title + str(counter)
                chunk = ' '.join(content_tokens[i:i+chunk_length])
                chunk_processed = self.__preprocess_content(
                    chunk, False, False, False)
                stemmed_chunk_processed = self.__preprocess_content(
                    chunk, False, True, False)
                entry = {
                    'name': chunk_name,
                    'content': chunk_processed,
                    'content_stemmed': stemmed_chunk_processed
                }
                corpus_chunks.append(entry)
                counter += 1

        return corpus_chunks
