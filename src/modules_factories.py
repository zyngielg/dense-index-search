from data.medqa_questions import MedQAQuestions
from retriever.retriever import Retriever
from retriever.ir_es import IR_ES
from retriever.REALM_like_retriever import REALM_like_retriever
from reader.reader import Reader
from reader.base_bert_reader import Base_BERT_Reader
from trainer.trainer import Trainer
from trainer.REALM_like_retriever_base_BERT_reader_trainer import REALM_like_retriever_base_BERT_reader_trainer
from trainer.ir_es_base_bert_trainer import IrEsBaseBertTrainer


class ReaderRetrieverFactory():
    def __init__(self, retriever_choice, reader_choice, from_es_session=False, load_weights=False) -> None:
        self.retriever_choice = retriever_choice
        self.reader_choice = reader_choice
        self.from_es_session = from_es_session
        self.load_weights = load_weights

    def create_retriever(self, ) -> Retriever:
        retriever = None

        if self.retriever_choice == "IR-ES":
            retriever = IR_ES(from_es_session=self.from_es_session)
        elif self.retriever_choice == "REALM-like":
            retriever = REALM_like_retriever(load_weights=self.load_weights)

        if retriever is None:
            print("Retriever has not been initialized. Check input arguments")
            quit()
        else:
            print(f"*** Initialized retriever {retriever.__class__.__name__} ***")
            return retriever

    def create_reader(self, load_weights=False) -> Reader:
        reader = None

        if self.reader_choice == "Base-BERT":
            reader = Base_BERT_Reader(load_weights=load_weights)

        if reader is None:
            print("Reader has not been initialized. Check input arguments")
            quit()
        else:
            print(f"*** Initialized reader {reader.__class__.__name__} ***")
            return reader


class TrainerFactory():
    def __init__(self, retriever: Retriever, reader: Reader, questions: MedQAQuestions, num_epochs: int, batch_size: int, lr: float) -> None:
        self.retriever = retriever
        self.reader = reader
        self.questions = questions
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

    def create_trainer(self) -> Trainer:
        trainer = None

        if type(self.retriever) == IR_ES and type(self.reader) == Base_BERT_Reader:
            trainer = IrEsBaseBertTrainer(
                self.questions, self.retriever, self.reader, self.num_epochs, self.batch_size, self.lr)
        elif type(self.retriever) == REALM_like_retriever and type(self.reader) == Base_BERT_Reader:
            trainer = REALM_like_retriever_base_BERT_reader_trainer(
                self.questions, self.retriever, self.reader, self.num_epochs, self.batch_size, self.lr)

        if trainer is None:
            print("Trainer has not been initialized. Check input arguments")
            quit()
        else:
            print(f"*** Initialized trainer {trainer.__class__.__name__} ***")
            return trainer