import json
import datetime

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

        self.__save_results(train_res, val_res, test_res)

    def __save_results(self, train_results, val_results, test_results):
        results = {
            "info": self.retriever.get_info(),
            "results": [train_results, val_results, test_results]
        }
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
        results_file = f"src/results/ir-es-based/{dt_string}__IR-ES__e2e.json"
        with open(results_file, 'w') as results_file:
            json.dump(results, results_file)
