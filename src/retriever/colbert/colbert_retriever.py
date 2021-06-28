import faiss
import torch
import numpy as np
import faiss

from data.medqa_corpus import MedQACorpus
from models.colbert import ColBERT
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize, sent_tokenize
from retriever.colbert.colbert_tokenizer import ColbertTokenizer
from retriever.colbert.colbert_score_calculator import ColBERTScoreCalculator
from retriever.retriever import Retriever
from tqdm import tqdm
from utils.pickle_utils import save_pickle, load_pickle
from utils.general_utils import torch_percentile, uniq, remove_duplicates_preserve_order


class ColBERTRetriever(Retriever):
    bioclinical_bert_name = "emilyalsentzer/Bio_ClinicalBERT"
    base_bert_name = "bert-base-uncased"

    base_weights_bio_path = "data/colbert/colbert-clinical-biobert-cosine-200000.dnn"
    base_weights_base_path = "data/colbert/colbert-base-bert-200000.dnn"    
    weights_file_directory = "src/results/colbert-based"
    weights_file_bio_name = "2021-06-24_18:01:03__ColBERT_e2e_retriever.pth"
    weights_file_base_name = "2021-06-24_18:01:03__ColBERT_e2e_retriever.pth"

    weights_file_bio_path = f"{weights_file_directory}/{weights_file_bio_name}"
    weights_file_base_path = f"{weights_file_directory}/{weights_file_base_name}"

    num_documents_faiss = 2000
    num_documents_reader = 5
    layers_to_not_freeze = ['3', '4', '5', '6',
                            '7', '8', '9', '10', '11', 'linear']

    stemmer = SnowballStemmer(language='english')

    chunk_100_unstemmed_path = "data/chunks_100_non_processed.pickle"
    chunk_150_unstemmed_path = "data/chunks_150_non_processed.pickle"
    bert_weights_path = "data/colbert-clinical-biobert-cosine-200000.dnn"

    faiss_index_path = "data/colbert/index_[colbert][clinicalbiobert200000][cosine-sim][chunks100unprocessed].index"
    document_embeddings_path = "data/colbert/document_embeddings_[colbert][clinicalbiobert200000][cosine-sim][chunks100unprocessed].pickle"
    document_embeddings_tensor_path = "data/colbert/document_embeddings_tensor.pt"
    embbedding2doc_id_path = "data/colbert/embbedding2doc_id_[colbert][clinicalbiobert200000][cosine-sim][chunks100unprocessed].pickle"
    doc_embeddings_lengths_path = "data/colbert/document_embeddings_lengths_[colbert][clinicalbiobert200000][cosine-sim][chunks100unprocessed].pickle"

    def __init__(self, load_weights=False, biobert_or_base_bert="bio") -> None:
        super().__init__()
        self.load_weights = load_weights
        self.biobert_or_base_bert = biobert_or_base_bert

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using {} device".format(self.device))

        # defining tokenizer and encoders

        # loading weights
        if biobert_or_base_bert == "bio":
            self.colbert = ColBERT.from_pretrained(self.bioclinical_bert_name,
                                                   query_maxlen=100,
                                                   doc_maxlen=180,
                                                   device=self.device,
                                                   dim=128)
            self.tokenizer = ColbertTokenizer(bert_name=self.bioclinical_bert_name,
                                              query_maxlen=self.colbert.query_maxlen,
                                              doc_maxlen=self.colbert.doc_maxlen)

            self.faiss_index_path = "data/colbert/index_[colbert][clinicalbiobert200000][cosine-sim][chunks100unprocessed].index"
            self.document_embeddings_path = "data/colbert/document_embeddings_[colbert][clinicalbiobert200000][cosine-sim][chunks100unprocessed].pickle"
            self.document_embeddings_tensor_path = "data/colbert/document_embeddings_tensor[clinicalbiobert200000].pt"
            self.embbedding2doc_id_path = "data/colbert/embbedding2doc_id_[colbert][clinicalbiobert200000][cosine-sim][chunks100unprocessed].pickle"
            self.doc_embeddings_lengths_path = "data/colbert/document_embeddings_lengths_[colbert][clinicalbiobert200000][cosine-sim][chunks100unprocessed].pickle"

            if load_weights:
                print(
                    f"Loading trained biobert weights from {self.weights_file_bio_path}")
                ColBERT.load_checkpoint(
                    self.weights_file_bio_path, self.colbert, False)
                self.q_encoder_weights_path = self.weights_file_bio_path
            else:
                print(
                    f"Loading base biobert weights from {self.base_weights_bio_path}")
                ColBERT.load_checkpoint(
                    self.base_weights_bio_path, self.colbert)
                self.base_weights_used = self.base_weights_bio_path
        else:
            self.colbert = ColBERT.from_pretrained(self.base_bert_name,
                                                   query_maxlen=100,
                                                   doc_maxlen=180,
                                                   device=self.device,
                                                   dim=128)
            self.tokenizer = ColbertTokenizer(bert_name=self.base_bert_name,
                                              query_maxlen=self.colbert.query_maxlen,
                                              doc_maxlen=self.colbert.doc_maxlen)

            self.faiss_index_path = "data/colbert/index_[colbert][basebert400000][l2-sim][chunks100unprocessed].index"
            self.document_embeddings_path = "data/colbert/document_embeddings_[colbert][basebert200000][cosine-sim][chunks100unprocessed].pickle"
            self.document_embeddings_tensor_path = "data/colbert/document_embeddings_tensor[basebert200000].pt"
            self.embbedding2doc_id_path = "data/colbert/embbedding2doc_id_[colbert][basebert200000][cosine-sim][chunks100unprocessed].pickle"
            self.doc_embeddings_lengths_path = "data/colbert/document_embeddings_lengths_[colbert][basebert200000][cosine-sim][chunks100unprocessed].pickle"

            if load_weights:
                print(
                    f"Loading trained biobert weights from {self.weights_file_base_path}")
                ColBERT.load_checkpoint(
                    self.weights_file_base_path, self.colbert, False)
                self.q_encoder_weights_path = self.weights_file_base_path
            else:
                print(
                    f"Loading base biobert weights from {self.base_weights_base_path}")
                ColBERT.load_checkpoint(
                    self.base_weights_base_path, self.colbert)
                self.base_weights_used = self.base_weights_base_path

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.colbert = torch.nn.DataParallel(self.colbert)

        self.freeze_layers()
        self.colbert.to(self.device)

        self.used_chunks_size = 100

    def query(self, input_ids, attention_mask):
        Q = self.colbert(input_ids, attention_mask=attention_mask)
        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        with torch.no_grad():
            D = self.colbert(input_ids=input_ids,
                             attention_mask=attention_mask)
            mask = torch.tensor(self.mask(input_ids),
                                device=D.device).unsqueeze(2).float()
            D = D * mask

            D = torch.nn.functional.normalize(D, p=2, dim=2)

            if not keep_dims:
                D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
                D = [d[mask[idx]] for idx, d in enumerate(D)]

            return D

    def score(self, Q, D):
        Q, D = self.query(*Q), self.doc(*D)
        if self.colbert.module.similarity_metric == 'cosine':
            x = (Q @ D.permute(0, 2, 1))
            return x.max(2).values.sum(1)
        else:  # l2
            return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.colbert.module.skiplist) and (x != 0) for x in d]
                for d in input_ids.cpu().tolist()]
        return mask

    def get_info(self):
        info = {}
        info['num documents faiss'] = self.num_documents_faiss
        info['num documents retrieved'] = self.num_documents_reader
        info['layers not to freeze'] = self.layers_to_not_freeze
        info['weights loaded'] = self.load_weights
        if self.load_weights:
            info['weights path'] = self.q_encoder_weights_path
        info["base_weights_type"] = self.biobert_or_base_bert
        info['index path'] = self.faiss_index_path
        info['chunk_size_used'] = self.used_chunks_size

        return info

    def retrieve_documents(self, queries: list):
        all_retrieved_docs, all_scores = [], []
        input_ids, mask = self.tokenizer.tensorize_queries(queries)
        Q = self.query(input_ids, mask)
        num_queries, embeddings_per_query, dim = Q.size()
        Q_faiss = Q.view(num_queries * embeddings_per_query,
                         dim).cpu().detach().float().numpy()
        _, embeddings_ids = self.index.search(
            Q_faiss, self.num_documents_faiss)

        embeddings_ids = torch.from_numpy(embeddings_ids)
        embeddings_ids = embeddings_ids.view(
            num_queries, embeddings_per_query * embeddings_ids.size(1))
        # embedding_ids_to_pids
        doc_ids = self.embbedding2doc_id[embeddings_ids].tolist()
        doc_ids = list(map(uniq, doc_ids))
        doc_ids_min = len(min(doc_ids, key=len))
        doc_ids_min = doc_ids_min if doc_ids_min < 1500 else 1500
        doc_ids = [x[:doc_ids_min] for x in doc_ids]

        # rank
        # .to(self.device).to(dtype=torch.float32)
        Q = Q.permute(0, 2, 1).contiguous().to(dtype=torch.float32)
        # preprocessing
        scores = self.score_calc.calculate_scores(Q, doc_ids)
        top_k = torch.topk(scores, self.num_documents_reader, dim=1)
        top_scores = top_k.values
        top_docs_ids = []
        for i in range(len(doc_ids)):
            top_docs_ids.append([doc_ids[i][x] for x in top_k.indices[i]])
        top_docs = []
        for i in range(len(top_docs_ids)):
            top_docs.append([self.documents[x] for x in top_docs_ids[i]])

        return top_docs, top_scores

    def freeze_layers(self):
        for name, param in self.colbert.named_parameters():
            if not any(x in name for x in self.layers_to_not_freeze):
                param.requires_grad = False

    def prepare_retriever(self, corpus: MedQACorpus = None, create_encodings=True, create_index=True):
        if self.used_chunks_size == 100:
            chunks_path = self.chunk_100_unstemmed_path
        elif self.used_chunks_size == 150:
            chunks_path = self.chunk_150_unstemmed_path

        if corpus is None:
            print(
                f"******** 0. Loading the corpus chunks of size {self.used_chunks_size} from {chunks_path} ********")
            self.corpus_chunks = load_pickle(chunks_path)
            print("********    ... chunks loaded ********")
        else:
            print(
                f"******** 0a. Generating corpus chunks of size {self.used_chunks_size} ... ********")
            self.corpus_chunks = self.__create_corpus_chunks(
                corpus=corpus.corpus, chunk_length=self.used_chunks_size)
            print("********    ... chunks generated ********")
            print(
                f"******** 0b. Saving corpus chunks at {chunks_path} ... ********")
            save_pickle(self.corpus_chunks, chunks_path)
            print("********    ... chunks saved ********")

        self.documents = [x['content'] for x in self.corpus_chunks]
        dimension = self.colbert.module.dim

        if create_encodings:
            self.colbert.eval()
            print("******** 1a. Creating document embeddings ... ********")
            embeddings = []
            self.doc_embeddings_lengths = []
            batch_size = 110
            for idx in tqdm(range(0, len(self.documents), batch_size)):
                contents = self.documents[idx:idx+batch_size]
                ids, mask = self.tokenizer.tensorize_documents(contents)
                with torch.no_grad():
                    batch_embeddings = self.doc(ids, mask, keep_dims=False)
                self.doc_embeddings_lengths += [d.size(0)
                                                for d in batch_embeddings]
                embeddings.append(torch.cat(batch_embeddings))

            save_pickle(self.doc_embeddings_lengths,
                        self.doc_embeddings_lengths_path)
            self.embeddings_tensor = torch.cat(embeddings)
            assert dimension == self.embeddings_tensor.shape[-1]

            self.doc_embeddings = self.embeddings_tensor.float().numpy()
            print("********     ... embeddings created *********")

            print("******** 1b. Creating embedding2doc_id matrix ... ********")
            total_num_embeddings = sum(self.doc_embeddings_lengths)
            self.embbedding2doc_id = np.zeros(total_num_embeddings, dtype=int)

            offset = 0
            for doc_id, doc_length in enumerate(self.doc_embeddings_lengths):
                self.embbedding2doc_id[offset: offset + doc_length] = doc_id
                offset += doc_length
            print("******** 1b. ... matrix created ... ********")

            print(
                "******** 1c. Saving document embeddings and embedding2doc_id to file ... ********")
            torch.save(self.embeddings_tensor,
                       self.document_embeddings_tensor_path)
            save_pickle(self.embbedding2doc_id, self.embbedding2doc_id_path)
            print("********     ... embeddings and matrix saved *********")
        else:
            print(
                "******** 1. Loading document embeddings and embedding2doc_id matrix... ********")
            self.embeddings_tensor = torch.load(
                self.document_embeddings_tensor_path)
            self.doc_embeddings = self.embeddings_tensor.float().numpy()
            self.doc_embeddings_lengths = load_pickle(
                self.doc_embeddings_lengths_path)
            self.embbedding2doc_id = load_pickle(self.embbedding2doc_id_path)
            print("********    ... embeddings and matrix loaded ********")

        if create_index:
            print("******** 2a. Creating and populating faiss index ...  *****")
            # num_embeddings = doc_embeddings.shape[0]
            # partitions = 1 << math.ceil(
            #     math.log2(8 * math.sqrt(num_embeddings)))
            sample = np.ascontiguousarray(self.doc_embeddings[0::20])
            # build a flat (CPU) index
            quantizer = faiss.IndexFlatIP(dimension)
            num_partitions = 1000
            index = faiss.IndexIVFPQ(
                quantizer, dimension, num_partitions, 16, 8)
            if self.device.type != 'cpu':
                # index = faiss.index_cpu_to_all_gpus(index)
                res = faiss.StandardGpuResources()  # use a single GPU
                index = faiss.index_cpu_to_gpu(res, 1, index)

            # index.train(sample)

            index.train(sample)
            index.add(self.doc_embeddings)
            print("training C")
            self.index = index
            print("********      ... index created and populated ********")

            print("******** 2b. Saving the index ... ********")
            if self.device.type != 'cpu':
                index = faiss.index_gpu_to_cpu(index)
            faiss.write_index(index, self.faiss_index_path)
            print("********     ... index saved ********")
        else:
            print("******** 2. Loading faiss index ...  ********")
            index = faiss.read_index(self.faiss_index_path)
            res = faiss.StandardGpuResources()  # use a single GPU
            self.index = faiss.index_cpu_to_gpu(res, 1, index)
            print("********    ... index loaded ********")

        print("******** 3. Preparing the ColBERT score calculator ********")
        self.score_calc = ColBERTScoreCalculator(
            doclens=self.doc_embeddings_lengths, embeddings_tensor=self.embeddings_tensor, device=self.device)

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
            window = 25
            counter = 0
            for i in range(0, len(content_tokens), window):
                chunk_name = title + str(counter)
                chunk = ' '.join(content_tokens[i:i+chunk_length]).replace(
                    ' ( ', ' ').replace(' ) ', ' ').replace(' [ ', ' ').replace(' ] ', ' ')
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
