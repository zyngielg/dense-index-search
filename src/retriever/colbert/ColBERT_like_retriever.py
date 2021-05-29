import faiss
import torch
import numpy as np
import faiss
import json
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
    
    bert_name = "vespa-engine/colbert-medium"
    # change to specify the weights file
    q_encoder_weights_path = ""
    num_documents = 5
    layers_to_not_freeze = ['9', '10', '11', 'pooler']
    
    stemmer = SnowballStemmer(language='english')

    faiss_index_path = "data/vespa-engine-colbert-medium_index_chunks_150_non_processed.index"
    document_encodings_path = "data/vespa-engine-colbert-medium_document_encodings_chunks_150_non_processed.pickle"
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
        self.colbert =  ColBERT.from_pretrained(self.bert_name,
                                                query_maxlen=512,
                                                doc_maxlen=512)
        self.tokenizer = ColbertTokenizer(bert_name=self.bert_name,
                                           query_maxlen=self.colbert.query_maxlen,
                                           doc_maxlen=self.colbert.query_maxlen)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.colbert = torch.nn.DataParallel(self.colbert)

        # loading weights
        if load_weights:
            self.colbert.load_state_dict(
                torch.load(self.colbert_weights_path))

        # loading index
        if load_index:
            self.index = faiss.read_index(self.faiss_index_path)

        # freezing layers
        self.freeze_layers()
        self.colbert.to(self.device)

        # info about used chunks
        self.used_chunks_size = 150

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

    def retrieve_documents(self, query: str):
        query_tokenized = self.tokenizer(query,
                                         add_special_tokens=True,
                                         max_length=512,
                                         padding='max_length',
                                         truncation=True,
                                         return_tensors="pt")
        query_embedding = self.q_encoder(
            **query_tokenized.to(self.device)).pooler_output.flatten()
        query_faiss_input = query_embedding.cpu().detach().reshape(1, 768).numpy()
        _, I = self.index.search(query_faiss_input, self.num_documents)

        retrieved_documents = [self.corpus_chunks[i] for i in I[0]]
        return retrieved_documents

    def freeze_layers(self):
        for name, param in self.colbert.named_parameters():
            if not any(x in name for x in self.layers_to_not_freeze):
                param.requires_grad = False
            else:
                print(
                    f"Layer {name} not frozen (status: {param.requires_grad})")

    def prepare_retriever(self, corpus: MedQACorpus = None, create_encodings=True, create_index=True):
        if self.load_index is False:
            if corpus is None:
                if self.used_chunks_size == 100:
                    chunks_path = self.chunk_100_unstemmed_path
                elif self.used_chunks_size == 150:
                    chunks_path = self.chunk_150_unstemmed_path
                print(f"Loading the corpus chunks of size {self.used_chunks_size} from {chunks_path}")
                self.corpus_chunks = load_pickle(chunks_path)
            else:
                self.corpus_chunks = self.__create_corpus_chunks(
                    corpus=corpus.corpus, chunk_length=self.used_chunks_size)
                save_pickle(self.corpus_chunks, self.chunk_150_unstemmed_path)

            num_docs = len(self.corpus_chunks)
            dimension = 512

            if create_encodings:
                print("******** 0a. Creating the chunks' input ids and attention masks ... ********")
                chunks_input_ids, chunks_attention_masks = self.tokenizer.tensorize_documents(self.corpus_chunks)
                print("******** ... completed ********")
                print("******** 1a. Creating the chunks' encodings ... ********")
                chunks_encodings = np.empty(
                    (num_docs, dimension)).astype('float32')
                for i in tqdm(range(len(chunks_input_ids))):
                    input_ids = chunks_input_ids[i]
                    attention_mask = chunks_attention_masks[i]
                    
                    encoding = self.colbert.doc(
                        input_ids.to(self.device), attention_mask.to(self.device))
                    chunks_encodings[i] = encoding

                print("********     ... chunks' encodings created ********")

                print("******** 1b. Saving chunk encodingx to file ... ********")
                save_pickle(chunks_encodings,
                            file_path=self.document_encodings_path)
                print("********     ... encodings saved *********")
            else:
                print("******** 1. Loading chunk encodings ... ********")
                chunks_encodings = load_pickle(self.document_encodings_path)
                print("********    ... encodings loaded ********")

            if create_index:
                print("******** 2a. Creating and populating faiss index ...  *****")
                # build a flat (CPU) index
                index = faiss.IndexFlatIP(dimension)
                if self.device.type != 'cpu':
                    res = faiss.StandardGpuResources()  # use a single GPU
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                index.train(chunks_encodings)
                index.add(chunks_encodings)
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
                print("********    ... index loaded ********")

        print("*** Finished Preparing the REALM-like retriever ***")

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
