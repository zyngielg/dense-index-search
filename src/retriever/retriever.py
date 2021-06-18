from abc import ABC, abstractmethod

from data.medqa_corpus import MedQACorpus


class Retriever(ABC):
    @abstractmethod
    def retrieve_documents(self, query: str):
        pass

    @abstractmethod
    def prepare_retriever(self, corpus: MedQACorpus = None, initialize=False, create_encodings=True, create_index=True):
        pass
    
    @abstractmethod
    def get_info(self):
        pass
