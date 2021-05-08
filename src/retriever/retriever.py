from abc import ABC, abstractmethod

from data.medqa_corpus import MedQACorpus


class Retriever(ABC):
    @abstractmethod
    def retrieve_documents(self, query: str):
        pass

    @abstractmethod
    def prepare_retriever(self, corpus: MedQACorpus = None):
        pass
