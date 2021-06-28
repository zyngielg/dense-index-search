from data.medqa_questions import MedQAQuestions
from retriever.base_bert_retriever import BaseBertRetriever
from retriever.colbert.colbert_retriever import ColBERTRetriever
from retriever.ir_es import IR_ES
from retriever.realm_like_retriever import REALMLikeRetriever
from retriever.retriever import Retriever
from reader.base_bert_reader import Base_BERT_Reader
from reader.bert_for_multiple_choice_reader import BERT_multiple_choice_reader
from reader.reader import Reader
from solution.colbert_e2e import ColBERTe2e
from solution.colbert_retriever_base_bert_reader import ColbertRetrieverBaseBertReader
from solution.base_bert_retriever_base_bert_reader import BaseBERTRetrieverBaseBERTReader
from solution.base_bert_retriever_bert_for_multiple_choice_reader import BaseBERTRetrieverBERTForMultipleChoiceReader
from solution.ir_es_base_bert import IrEsBaseBert
from solution.ir_es_e2e import IrEse2e
from solution.realm_like_retriever_base_bert_reader import REALMLikeRetrieverBaseBERTReader
from solution.solution import Solution


class ReaderRetrieverFactory():
    def __init__(self, retriever_choice, reader_choice, from_es_session=False, load_weights=False, load_index=False, colbert_base="bio") -> None:
        self.retriever_choice = retriever_choice
        self.reader_choice = reader_choice
        self.from_es_session = from_es_session
        self.load_weights = load_weights
        self.load_index = load_index
        self.colbert_base = colbert_base

    def create_retriever(self, ) -> Retriever:
        retriever = None

        if self.retriever_choice == "IR-ES":
            retriever = IR_ES(from_es_session=self.from_es_session)
        elif self.retriever_choice == "Base-BERT":
            retriever = BaseBertRetriever(load_weights=self.load_weights)
        elif self.retriever_choice == "REALM-like":
            retriever = REALMLikeRetriever(load_weights=self.load_weights)
        elif self.retriever_choice == "ColBERT":
            retriever = ColBERTRetriever(
                load_weights=self.load_weights, biobert_or_base_bert=self.colbert_base)

        if retriever is None:
            print("Retriever has not been initialized. Check input arguments")
            quit()
        else:
            print(
                f"*** Initialized retriever {retriever.__class__.__name__} ***")
            return retriever

    def create_reader(self) -> Reader:
        reader = None

        if self.reader_choice == "Base-BERT":
            reader = Base_BERT_Reader(load_weights=self.load_weights)
        elif self.reader_choice == "BERT-for-multiple-choice":
            reader = BERT_multiple_choice_reader(load_weights=self.load_weights)

        if reader is None:
            print("Reader has not been initialized. Check input arguments")
        else:
            print(f"*** Initialized reader {reader.__class__.__name__} ***")
            return reader


class SolutionFactory():
    def __init__(self, retriever: Retriever, reader: Reader, questions: MedQAQuestions, num_epochs: int, batch_size: int, lr: float) -> None:
        self.retriever = retriever
        self.reader = reader
        self.questions = questions
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

    def create_solution(self) -> Solution:
        solution = None

        if type(self.retriever) == IR_ES:
            if type(self.reader) == Base_BERT_Reader:
                solution = IrEsBaseBert(
                    self.questions, self.retriever, self.reader, self.num_epochs, self.batch_size, self.lr)
            else:
                solution = IrEse2e(self.questions, self.retriever,
                                   self.reader, self.num_epochs, self.batch_size, self.lr)
        elif type(self.retriever) == BaseBertRetriever:
            if type(self.reader) == Base_BERT_Reader:
                solution = BaseBERTRetrieverBaseBERTReader(
                    self.questions, self.retriever, self.reader, self.num_epochs, self.batch_size, self.lr)
            elif type(self.reader) == BERT_multiple_choice_reader:
                solution = BaseBERTRetrieverBERTForMultipleChoiceReader(
                    self.questions, self.retriever, self.reader, self.num_epochs, self.batch_size, self.lr)
        elif type(self.retriever) == REALMLikeRetriever:
            solution = REALMLikeRetrieverBaseBERTReader(
                self.questions, self.retriever, self.reader, self.num_epochs, self.batch_size, self.lr)
        elif type(self.retriever) == ColBERTRetriever:
            if type(self.reader) == Base_BERT_Reader:
                solution = ColbertRetrieverBaseBertReader(
                    self.questions, self.retriever, self.reader, self.num_epochs, self.batch_size, self.lr)
            else:
                solution = ColBERTe2e(
                    self.questions, self.retriever, self.num_epochs, self.batch_size, self.lr)

        if solution is None:
            print("Solution has not been initialized. Check input arguments")
            quit()
        else:
            print(
                f"*** Initialized solution {solution.__class__.__name__} ***")
            return solution
