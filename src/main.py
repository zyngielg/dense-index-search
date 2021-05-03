import argparse
from retriever.retriever import Retriever
from retriever.ir_es import IR_ES
from reader.reader import Reader
from data.medqa_questions import MedQAQuestions
from reader.base_bert_reader import Base_BERT_Reader

from trainer.ir_es_base_bert_trainer import IrEsBaseBertTrainer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', help="Choose mode. Possible options: TRAINING, QA")
    parser.add_argument(
        "corpus", help="Choose dataset used as the context corpus. Possible options: MedQA, FindZebra")
    parser.add_argument(
        'retriever', help="Choose retriever to be used. Possible options: IR-ES, IR-CUSTOM, ColBERT, DPR")
    parser.add_argument(
        'reader', help="Choose reader to be used. Possible options: Base-BERT")
    return parser.parse_args()


def choose_retriever_and_reader(retriever_choice: str, reader_choice: str):
    if retriever_choice == 'IR-ES':
        retriever = IR_ES()
        print("Chosen retriever: IR-ES")
    else:
        retriever = None

    if reader_choice == 'Base-BERT':
        reader = Base_BERT_Reader(load_weights=False)
        print("Chosen reader: Base-BERT")
    else:
        reader = None

    if reader is None and retriever is None:
        print("Retriever and reader have not been initialized. Check input arguments")
        quit()

    return retriever, reader


def qa(questions, retriever: Retriever, reader: Reader):
    if type(retriever) == IR_ES:
        retriever.__class__ = IR_ES
        if reader == None:
            print('Running IR-ES module e2e (val set)')
            retriever.run_ir_es_e2e(questions.questions_val)
        elif type(reader) == Base_BERT_Reader:
            print('nice')


def train(questions: MedQAQuestions, retriever: Retriever, reader: Reader):
    if type(retriever) == IR_ES and type(reader) == Base_BERT_Reader:
        trainer = IrEsBaseBertTrainer(
            questions=questions, retriever=retriever, reader=reader)
        trainer.train()


if __name__ == "__main__":
    args = parse_arguments()
    medqa_questions = MedQAQuestions(stemming=False)
    retriever, reader = choose_retriever_and_reader(
        args.retriever, args.reader)

    if args.mode == "QA":
        qa(questions=medqa_questions.questions_dev,
           retriever=retriever, reader=reader)
    elif args.mode == "TRAINING":
        train(medqa_questions, retriever, reader)
