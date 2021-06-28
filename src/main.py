import argparse
from data.medqa_corpus import MedQACorpus
from modules_factories import ReaderRetrieverFactory, SolutionFactory
from retriever.retriever import Retriever
from retriever.ir_es import IR_ES
from reader.reader import Reader
from data.medqa_questions import MedQAQuestions
from reader.base_bert_reader import Base_BERT_Reader

# TODO: move to separate config file
num_epochs = 4
batch_size = 32
lr = 5e-5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', help="Choose mode. Possible options: TRAINING, QA")
    parser.add_argument(
        'retriever', help="Choose retriever to be used. Possible options: IR-ES, Base-BERT, REALM-like")
    parser.add_argument(
        'reader', help="Choose reader to be used. Possible options: Base-BERT, BERT-for-multiple-choice")
    parser.add_argument("--questions_filtered",
                        dest='questions_filtered', default=False, action='store_true')
    parser.add_argument(
        '--colbert_base', dest="colbert_base", help="Choise of BERT model for the ColBERT retrieval. Possible options: bio, base")
    parser.add_argument('--batch_size', dest="batch_size",
                        type=int, default=32, help="Batch size")
    parser.add_argument('--num_epochs', dest="num_epochs",
                        type=int, default=4, help="Number of epochs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    medqa_questions = MedQAQuestions(args.questions_filtered, stemming=False)
    medqa_corpus = MedQACorpus(stemming=False)

    retriever_reader_factory = ReaderRetrieverFactory(
        retriever_choice=args.retriever, reader_choice=args.reader, colbert_base=args.colbert_base)
    retriever = retriever_reader_factory.create_retriever()
    retriever.prepare_retriever(
        corpus=None, create_encodings=False, create_index=False)
    reader = retriever_reader_factory.create_reader()

    solution_factory = SolutionFactory(retriever=retriever, reader=reader, questions=medqa_questions,
                                       num_epochs=args.num_epochs, batch_size=args.batch_size, lr=lr)
    solution = solution_factory.create_solution()

    if args.mode == "QA":
        solution.qa()
    elif args.mode == "TRAINING":
        solution.train()
