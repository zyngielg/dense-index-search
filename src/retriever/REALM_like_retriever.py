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
from utils.pickle_utils import save_pickle


class REALM_like_retriever(Retriever):
    tokenizer_type = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    d_encoder_bert_type = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    q_encoder_bert_type = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    stemmer = SnowballStemmer(language='english')
    # change to specify the weights file
    q_encoder_weights_path = ""
    faiss_index_path = "data/medqa/textbooks/chunks_100_non_processed.index"
    document_encodings_path = "data/medqa/textbooks/encodings.pickle"

    def __init__(self, load_weights=False, load_index=False) -> None:
        super().__init__()
        self.load_index = load_index
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using {} device".format(self.device))

        # defining tokenizer and encoders
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_type)
        self.d_encoder = AutoModel.from_pretrained(self.d_encoder_bert_type)
        self.q_encoder = AutoModel.from_pretrained(self.q_encoder_bert_type)

        # loading weights
        if load_weights:
            self.q_encoder.load_state_dict(
                torch.load(self.q_encoder_weights_path))

        # loading index
        if load_index:
            self.index = faiss.read_index(self.faiss_index_path)

        # freezing layers
        self.freeze_layers(['pooler'])

    def retrieve_documents(self, query: str):
        return super().retrieve_documents(query)

    def freeze_layers(self, q_encoder_layers_to_not_freeze):
        for name, param in self.d_encoder.named_parameters():
            param.requires_grad = False

        for name, param in self.q_encoder.named_parameters():
            if not any(x in name for x in q_encoder_layers_to_not_freeze):
                param.requires_grad = False
            else:
                print(f"Layer {name} not frozen")

    def prepare_retriever(self, corpus: MedQACorpus = None):
        if self.load_index is False:
            if corpus is None:
                print(
                    "The corpus has not been properly initialized. Check input arguments")
                quit()

            chunk_length = 100
            corpus_chunks = self.__create_corpus_chunks(
                corpus=corpus.corpus, chunk_length=chunk_length)

            num_docs = len(corpus_chunks)
            dimension = 768

            print("******** 1. Creating the chunks' encodings ********")
            chunks_encodings = np.empty(
                (num_docs, dimension)).astype('float32')
            for idx, chunk in enumerate(tqdm(corpus_chunks)):
                content = chunk['content']
                content_tokenized = self.tokenizer(content,                                              add_special_tokens=True,
                                                   max_length=512,
                                                   padding='max_length',
                                                   truncation=True,
                                                   return_tensors="pt")
                encoding = self.d_encoder(**content_tokenized.to(self.device))
                encoding = encoding.pooler_output.flatten().cpu().detach()
                chunks_encodings[idx] = encoding

            print("********    Chunks' encodings created ********")

            print("******** 2. Creating faiss index and loading the encodings *****")
            index = faiss.IndexFlatIP(dimension)  # build a flat (CPU) index
            if self.device != 'cpu':
                res = faiss.StandardGpuResources()  # use a single GPU
                index = faiss.index_cpu_to_gpu(res, 0, index)
            index.train(chunks_encodings)
            index.add(chunks_encodings)
            print("********    Index created and populated ********")

            print("******** 3. Saving the embeddings and the index to the file *********")
            save_pickle(chunks_encodings,
                        file_path=self.document_encodings_path)
            print("********    Encodings saved ********")

            if self.device != 'cpu':
                index = faiss.index_gpu_to_cpu(index)
            faiss.write_index(
                index, "data/five_random_quotes_size_30000.index")
            print("********    Index saved *********")
        else:
            print(
                f"******** Loading index from the file {self.faiss_index_path} ********")

            print(f"******** Index loaded ********")

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
