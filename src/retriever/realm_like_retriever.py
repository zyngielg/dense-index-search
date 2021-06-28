from pathlib import WindowsPath
import faiss
import torch
import numpy as np
import faiss
from data.medqa_corpus import MedQACorpus
from retriever.retriever import Retriever
from transformers import BertTokenizerFast, BertConfig
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
from utils.pickle_utils import save_pickle, load_pickle
from utils.convert_tf_to_pytorch import load_tf_weights_in_bert
from models.realm_embedder import REALMEmbedder


class REALMLikeRetriever(Retriever):
    bert_type = "bert-base-uncased"
    base_weights_path = "data/realm/base-realm-embedder.pt"

    weights_file_directory = "src/results/realm-based"
    weights_file_name = "2021-06-21_14:49:56__REALM_retriever.pth"
    saved_weights_path = f"{weights_file_directory}/{weights_file_name}"

    vocab_file_path = "data/realm-tf-to-pytorch/assets/vocab.txt"
    downloaded_model_path = "data/realm-tf-to-pytorch/"

    num_documents = 4
    layers_to_not_freeze = ['6', '7', '8', '9', '10', '11', 'pooler', 'prediction']

    faiss_index_path = "data/realm/index.index"
    document_encodings_path = "data/realm/document_embeddings_chunks_100_unprocessed.pickle"
    chunk_150_unstemmed_path = "data/chunks_150_non_processed.pickle"
    chunk_100_unstemmed_path = "data/chunks_100_non_processed.pickle"

    def __init__(self, load_weights=False) -> None:
        super().__init__()
        self.load_weights = load_weights
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using {} device".format(self.device))

        self.tokenizer = BertTokenizerFast(self.vocab_file_path)
        # document embedder
        self.d_embedder = REALMEmbedder(BertConfig())        
        try:
            self.d_embedder.load_state_dict(
                torch.load(self.base_weights_path))
            print(
                f"[d_embedder] Loaded base model weights from {self.base_weights_path}")
        except:
            self.d_embedder = load_tf_weights_in_bert(
                REALMEmbedder(BertConfig()), self.downloaded_model_path)
            torch.save(self.d_embedder.state_dict(), self.base_weights_path)
            print("Initialized base model from the downloaded tensorflow checkpoint")
        
        self.q_embedder = REALMEmbedder(BertConfig())
        if load_weights:
            print(
                f"[q_embedder] Loading saved model weights from {self.saved_weights_path}")
            saved_model = torch.load(self.saved_weights_path)
            saved_model = {key.replace("module.", ""): value for key, value in saved_model.items()}
            self.q_embedder.load_state_dict(saved_model)
        else:
            self.q_embedder.load_state_dict(torch.load(self.base_weights_path))
            print(
                f"[q_embedder] Loaded base model weights from {self.base_weights_path}")
        self.freeze_layers()
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            if torch.cuda.device_count() == 8:
                self.d_embedder = torch.nn.DataParallel(self.d_embedder, device_ids=[0,1,2,3,4])
                self.q_embedder = torch.nn.DataParallel(self.q_embedder, device_ids=[0,1,2,3,4])
            else:
                self.d_embedder = torch.nn.DataParallel(self.d_embedder)
                self.q_embedder = torch.nn.DataParallel(self.q_embedder)
        self.d_embedder.to(self.device)
        self.q_embedder.to(self.device)
        self.used_chunks_size = 100

    def get_info(self):
        info = {}
        info['num documents retrieved'] = self.num_documents

        info['bert used'] = self.bert_type
        info['layers not to freeze'] = self.layers_to_not_freeze
        if self.load_weights:
            info['weights path'] = self.saved_weights_path
        info['index path'] = self.faiss_index_path
        info['chunk_size_used'] = self.used_chunks_size

        return info

    def retrieve_documents(self, queries: list):
        query_tokenized = self.tokenizer(queries,
                                         add_special_tokens=True,
                                         max_length=512,
                                         padding='max_length',
                                         truncation=True,
                                         return_tensors="pt")
        queries_embedding = self.q_embedder(**query_tokenized)
        queries_faiss_input = queries_embedding.cpu().detach().numpy()
        scores, doc_ids = self.index.search(
            queries_faiss_input, self.num_documents)

        retrieved_documents = []
        for id_list in doc_ids:
            retrieved_documents.append(
                [self.corpus_chunks[i]['content'] for i in id_list])

        retrieved_documents_score_calc = [self.corpus_chunks[i]['content']
                                          for x in doc_ids for i in x]

        docs_tokenized = self.tokenizer(retrieved_documents_score_calc,
                                        add_special_tokens=True,
                                        max_length=512,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors="pt")
       
        with torch.no_grad():
            docs_embeddings = self.d_embedder(**docs_tokenized)
        docs_embeddings = docs_embeddings.view(4, 4, -1)
        recalculated_scores = torch.einsum(
            "bd,bcd->bc", queries_embedding, docs_embeddings)
        recalculated_scores = torch.transpose(recalculated_scores, 0, 1)

        return recalculated_scores, retrieved_documents

    def freeze_layers(self):
        for name, param in self.d_embedder.named_parameters():
            param.requires_grad = False

        for name, param in self.q_embedder.named_parameters():
            if not any(x in name for x in self.layers_to_not_freeze):
                param.requires_grad = False

    def prepare_retriever(self, corpus: MedQACorpus = None, create_encodings=True, create_index=True):
        if self.used_chunks_size == 100:
            chunks_path = self.chunk_100_unstemmed_path
        elif self.used_chunks_size == 150:
            chunks_path = self.chunk_150_unstemmed_path

        if corpus:
            self.corpus_chunks = self.__create_corpus_chunks(
                corpus=corpus.corpus, chunk_length=self.used_chunks_size)
            save_pickle(self.corpus_chunks, chunks_path)
        else:
            print(f"Loading the corpus from {chunks_path}")
            self.corpus_chunks = load_pickle(chunks_path)

        num_docs = len(self.corpus_chunks)
        dimension = 128

        if create_encodings:
            self.__generate_embeddings(num_docs, dimension)
        else:
            print("******** 1. Loading chunk encodings ... ********")
            self.document_embeddings = load_pickle(
                self.document_encodings_path)
            print("********    ... encodings loaded ********")

        if create_index:
            print("******** 2a. Creating and populating faiss index ...  *****")
            # build a flat (CPU) index
            index = faiss.IndexFlatIP(dimension)
            if self.device.type != 'cpu':
                res = faiss.StandardGpuResources()  # use a single GPU
                index = faiss.index_cpu_to_gpu(res, 0, index)
            index.add(self.document_embeddings)
            self.index = index
            print("********      ... index created and populated ********")

            print("******** 2b. Saving the index ... ********")
            if self.device.type != 'cpu':
                index = faiss.index_gpu_to_cpu(index)
            faiss.write_index(index, self.faiss_index_path)
            print("********     ... index saved ********")
        else:
            print("******** 2. Loading faiss index ...  *****")
            self.index = faiss.read_index(self.faiss_index_path)
            res = faiss.StandardGpuResources()  # use a single GPU
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("********    ... index loaded ********")

        print("*** Finished Preparing the REALM-like retriever ***")

    def __generate_embeddings(self, num_docs, dimension):
        print("******** 1a. Creating the chunks' embeddings ... ********")
        self.document_embeddings = np.empty(
            (num_docs, dimension)).astype('float32')
        batch_size = 110
        self.d_embedder.eval()
        for i in tqdm(range(0, len(self.corpus_chunks), batch_size)):
            # , chunk in enumerate(tqdm(self.corpus_chunks)):
            chunks = self.corpus_chunks[i:i+batch_size]
            contents = [chunk['content'] for chunk in chunks]
            content_tokenized = self.tokenizer(contents,
                                               add_special_tokens=True,
                                               max_length=512,
                                               padding='max_length',
                                               truncation=True,
                                               return_tensors="pt")
            with torch.no_grad():
                embeddings = self.d_embedder(**content_tokenized)
            self.document_embeddings[i:i + batch_size] = embeddings.cpu()

        print("********     ... chunks' embeddings created ********")

        print("******** 1b. Saving chunk embeddings to file ... ********")
        save_pickle(self.document_embeddings,
                    file_path=self.document_encodings_path)
        print("********     ... embeddings saved *********")

    def __preprocess_content(self, content, remove_stopwords, stemming, remove_punctuation):
        if not remove_stopwords and not stemming and not remove_punctuation:
            return content.lower().strip()
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
