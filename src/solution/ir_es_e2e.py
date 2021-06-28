from data.medqa_questions import MedQAQuestions
from reader.reader import Reader
from retriever.retriever import Retriever
from solution.solution import Solution


class IrEse2e(Solution):
    def __init__(self, questions: MedQAQuestions, retriever: Retriever, reader: Reader, num_epochs: int, batch_size: int, lr: float) -> None:
        super().__init__(questions, retriever, reader, num_epochs, batch_size, lr)

    def train(self):
        print("IR-ES e2e is not a trainable solution.")
        quit()

    def qa(self):
        print('********* Running IR-ES QA e2e *********')
        print("****** Training set ******")
        train_res = self.retriever.run_ir_es_e2e(self.questions_train,
                                                 doc_flag=0)
        print("****** Validation set ******")
        val_res = self.retriever.run_ir_es_e2e(questions=self.questions_val,
                                               doc_flag=1)
        print("****** Test set ******")
        test_res = self.retriever.run_ir_es_e2e(self.questions_test,
                                                doc_flag=2)

        self.retriever.save_results(train_res, val_res, test_res)
