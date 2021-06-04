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
    document_encodings_path = "data/document_encodings_colbert_l2_clinical_bert_chunks_100_non_processed.pickle"
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
            encodings = []
            for idx, chunk in enumerate(tqdm(self.corpus_chunks)):
                content = chunk['content']

                ids, mask = self.tokenizer.tensorize_documents([content])
                test = self.tokenizer.doc_tokenizer.decode(ids[0])
                with torch.no_grad():
                    encoding = self.colbert.module.doc(ids, mask)[0]
                encodings.append(encoding)

            encodings = torch.cat(encodings)
            assert dimension == encodings.shape[-1]

            chunks_encodings = encodings.float().numpy()

            print("******** 1b. Saving chunk encodingx to file ... ********")
            save_pickle(chunks_encodings,
                        file_path=self.document_encodings_path)
            print("********     ... encodings saved *********")

        else:
            print("******** 1. Loading chunk encodings ... ********")
            chunks_encodings = load_pickle(self.document_encodings_path)
            print("********    ... encodings loaded ********")
        #     documents = [x['content'] for x in self.corpus_chunks]
        #     print(
        #         "******** 0a. Creating the chunks' input ids and attention masks ... ********")
        #     batch_size = 32
        #     batches, reverse_indices = self.tokenizer.tensorize_documents(documents, bsize=batch_size)
        #     print("******** ... completed ********")

        #     print("******** 1a. Creating the chunks' encodings ... ********")
        #     chunks_encodings_list = []
        #     for input_ids, attention_mask in tqdm(batches):
        #         with torch.no_grad():
        #             encoding = self.colbert.module.doc(input_ids, attention_mask)
        #             D = [d for batch in encoding for d in batch]
        #             # D = [D[idx] for idx in reverse_indices.tolist()]
        #         chunks_encodings_list.append(encoding)

        #     D = [d for batch in chunks_encodings_list for d in batch]
        #     # D = [D[idx] for idx in reverse_indices.tolist()]
        #     x = "and now what"
        #     D = torch.cat(D)
        #     torch.save(D, self.document_encodings_path)

        #     y = torch.load(self.document_encodings_path)
        #     print(type(y))
        #     print(y == D)
        #     quit()
        #     print("********     ... chunks' encodings created ********")

        # if create_encodings:
        #     documents = [x['content'] for x in self.corpus_chunks]
        #     print(
        #         "******** 0a. Creating the chunks' input ids and attention masks ... ********")
        #     chunks_input_ids, chunks_attention_masks = self.tokenizer.tensorize_documents(documents[:10])
        #     print("******** ... completed ********")
        #     print("******** 1a. Creating the chunks' encodings ... ********")
        #     chunks_encodings = np.empty(
        #         (num_docs, dimension)).astype('float32')

        #     chunks_encodings_list = []
        #     for i in tqdm(range(10)): #tqdm(range(len(chunks_input_ids))):
        #         input_ids = chunks_input_ids[i].unsqueeze(0)
        #         attention_mask = chunks_attention_masks[i].unsqueeze(0)
        #         with torch.no_grad():
        #             encoding = self.colbert.module.doc(input_ids, attention_mask)
        #             encoding = encoding.cpu()
        #             print(encoding.shape)
        #         # chunks_encodings[i] = encoding
        #         # print(chunks_encodings[i])
        #         chunks_encodings_list.append(encoding)
        #         print(chunks_encodings_list)
        #     chunks_encodings = torch.cat(chunks_encodings)
        #     print("********     ... chunks' encodings created ********")

            # print("******** 1b. Saving chunk encodingx to file ... ********")
            # save_pickle(chunks_encodings,
            #             file_path=self.document_encodings_path)
            # print("********     ... encodings saved *********")
        

        if create_index:
            print("******** 2a. Creating and populating faiss index ...  *****")
            # build a flat (CPU) index
            num_embeddings = chunks_encodings.shape[0]
            partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
            index = faiss.IndexFlatIP(dimension)
            if self.device.type != 'cpu':
                # index = faiss.index_cpu_to_all_gpus(index)
                res = faiss.StandardGpuResources()  # use a single GPU
                index = faiss.index_cpu_to_gpu(res, 1, index)
            # index.train(chunks_encodings)
            # index.add(chunks_encodings)
            # self.index = index

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
