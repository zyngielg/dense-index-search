import argparse
from data.medqa_corpus import MedQACorpus
from modules_factories import ReaderRetrieverFactory, TrainerFactory
from retriever.retriever import Retriever
from retriever.ir_es import IR_ES
from reader.reader import Reader
from data.medqa_questions import MedQAQuestions
from reader.base_bert_reader import Base_BERT_Reader

# TODO: move to separate config file
num_epochs = 3
batch_size = 16
lr = 5e-5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', help="Choose mode. Possible options: TRAINING, QA")
    parser.add_argument(
        "corpus", help="Choose dataset used as the context corpus. Possible options: MedQA, FindZebra")
    parser.add_argument(
        'retriever', help="Choose retriever to be used. Possible options: IR-ES, REALM-like")
    parser.add_argument(
        'reader', help="Choose reader to be used. Possible options: Base-BERT")
    return parser.parse_args()


def qa(questions, retriever: Retriever, reader: Reader):
    if type(retriever) == IR_ES:
        retriever.__class__ = IR_ES
        if reader == None:
            print('Running IR-ES module e2e (val set)')
            retriever.run_ir_es_e2e(questions.questions_val)
        elif type(reader) == Base_BERT_Reader:
            print('nice')


if __name__ == "__main__":
    args = parse_arguments()
    medqa_questions = MedQAQuestions(stemming=False)
    medqa_corpus = MedQACorpus(stemming=False)

    retriever_reader_factory = ReaderRetrieverFactory(
        retriever_choice=args.retriever, reader_choice=args.reader)
    retriever = retriever_reader_factory.create_retriever()
    retriever.prepare_retriever(medqa_corpus)
    reader = retriever_reader_factory.create_reader()

    if args.mode == "QA":
        qa(questions=medqa_questions.questions_dev,
           retriever=retriever, reader=reader)
    elif args.mode == "TRAINING":
        trainer_factory = TrainerFactory(retriever=retriever, reader=reader,
                                         questions=medqa_questions, num_epochs=num_epochs, batch_size=batch_size, lr=lr)
        trainer = trainer_factory.create_trainer()
        trainer.train()
