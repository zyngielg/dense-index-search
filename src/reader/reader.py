from abc import ABC, abstractmethod


class Reader(ABC):
    @abstractmethod
    def create_context(self):
        pass

    @abstractmethod
    def choose_answer(self, query, context, question_data):
        pass
    
    @abstractmethod
    def get_info(self):
        pass
