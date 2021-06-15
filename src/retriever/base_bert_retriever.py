import faiss
import torch
import numpy as np
import faiss
from data.medqa_corpus import MedQACorpus
from retriever.retriever import Retriever
from transformers import AutoTokenizer, AutoModel
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
from utils.pickle_utils import save_pickle, load_pickle


class BaseBertRetriever(Retriever):
    # "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    tokenizer_type = "emilyalsentzer/Bio_ClinicalBERT"
    d_encoder_bert_type = "emilyalsentzer/Bio_ClinicalBERT"
    q_encoder_bert_type = "emilyalsentzer/Bio_ClinicalBERT"
    stemmer = SnowballStemmer(language='english')
    # change to specify the weights file
    q_encoder_weights_path = ""
    num_documents = 5
    q_encoder_layers_to_not_freeze = ['8', '9', '10', '11', 'pooler']

    # faiss_index_path = "data/index_chunks_150_non_processed.index"
    faiss_index_path = "data/index_clinical_biobert_chunks_100_non_processed.index"
    # "data/document_encodings_chunks_150_non_processed.pickle"
    document_encodings_path = "data/document_embeddings_realm_clinical_bert_chunks_100_nonprocessed.pickle"
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_type)
        self.d_encoder = AutoModel.from_pretrained(self.d_encoder_bert_type)
        self.q_encoder = AutoModel.from_pretrained(self.q_encoder_bert_type)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.d_encoder = torch.nn.DataParallel(self.d_encoder)
            self.q_encoder = torch.nn.DataParallel(self.q_encoder)

        # loading weights
        if load_weights:
            self.q_encoder.load_state_dict(
                torch.load(self.q_encoder_weights_path))

        # loading index
        if load_index:
            self.index = faiss.read_index(self.faiss_index_path)

        # freezing layers
        self.freeze_layers()
        self.d_encoder.to(self.device)
        self.q_encoder.to(self.device)

        # info about used chunks
        self.used_chunks_size = 100

    def get_info(self):
        info = {}
        info['num documents retrieved'] = self.num_documents

        info['q_encoder'] = self.q_encoder_bert_type
        info['layers not to freeze'] = self.q_encoder_layers_to_not_freeze
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

        retrieved_documents = [self.corpus_chunks[i] for i in I[0] if i < len(self.corpus_chunks)]
        return retrieved_documents

    def freeze_layers(self):
        for name, param in self.d_encoder.named_parameters():
            param.requires_grad = False

        for name, param in self.q_encoder.named_parameters():
            if not any(x in name for x in self.q_encoder_layers_to_not_freeze):
                param.requires_grad = False
            # else:
            #     print(
            #         f"Layer {name} not frozen (status: {param.requires_grad})")

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
        dimension = 768

        if create_encodings:
            print("******** 1a. Creating the chunks' encodings ... ********")
            chunks_encodings = np.empty(
                (num_docs, dimension)).astype('float32')
            for idx, chunk in enumerate(tqdm(self.corpus_chunks)):
                content = chunk['content']
                content_tokenized = self.tokenizer(content,
                                                    add_special_tokens=True,
                                                    max_length=512,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_tensors="pt")
                encoding = self.d_encoder(
                    **content_tokenized.to(self.device))
                encoding = encoding.pooler_output.flatten().cpu().detach()
                chunks_encodings[idx] = encoding

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
