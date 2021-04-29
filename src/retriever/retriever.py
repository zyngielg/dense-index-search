from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def retrieve_documents(self, query: str):
        pass
