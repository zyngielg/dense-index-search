import argparse
from retriever.ir_es import IR_ES
from data.medqa_questions import MedQAQuestions
from reader.base_bert_reader import Base_BERT_Reader
from tqdm import tqdm
import torch
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', help="Choose mode. Possible options: TRAINING, QA")
    parser.add_argument(
        "corpus", help="Choose dataset used as the context corpus. Possible options: MedQA, FindZebra")
    parser.add_argument(
        'retriever', help="Choose retriever to be used. Possible options: IR-ES, IR-CUSTOM, ColBERT, DPR")
    parser.add_argument(
        'reader', help="Choose reader to be used. Possible options: Base_BERT")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    medqa_questions = MedQAQuestions(stemming=False)

    if args.retriever == 'IR-ES':
        retriever = IR_ES()
        retriever.setup_elasticsearch()
    else:
        retriever = None
    
    if args.reader == 'Base_BERT':
        reader = Base_BERT_Reader()
    else:
        reader = None

    if reader is None and retriever is None:
        print("Retriever and reader have not been initialized properly. Check input arguments")
    
    
    if type(retriever == IR_ES) and reader == None:
        retriever.__class__ = IR_ES
        retriever.run_ir_es_e2e(medqa_questions.questions_dev)
    
    # correct = 0
    # incorrect = 0
    
    # correct_predictions = []
    
    
    
    # for question_id, question_data in tqdm(medqa_questions.questions_dev.items()):
    #     question = question_data['question']
    #     correct_answer = question_data['answer']

    #     retrieved_documents = retriever.retrieve_documents(question)
    #     context = ' '.join(retrieved_documents)

    #     answer_outputs = []
    #     for option_letter, option in question_data['options'].items():
    #         qa = question + " " + option
    #         query = reader.tokenizer(context, qa, 
    #                                  add_special_tokens=True,
    #                                  max_length = 512, 
    #                                  padding='max_length',
    #                                  truncation=True,
    #                                  return_tensors="pt"
    #                                  )
    #         input_ids = query["input_ids"]
    #         token_type_ids = query["token_type_ids"]
    #         attention_mask = query["attention_mask"]

    #         with torch.no_grad():
    #             output = reader.model(input_ids=input_ids, 
    #                         attention_mask=token_type_ids,
    #                         token_type_ids=attention_mask)
    #         answer_outputs.append(output)

    #     answer_outputs = torch.stack(answer_outputs).transpose(0,1)
    #     answer_probabilities = reader.model.softmax(answer_outputs)
        
    #     prediction_idx = int(np.argmax(answer_probabilities, axis=1))
    #     prediction = question_data['options'][chr(prediction_idx + 65)]
        
    #     if prediction == correct_answer:
    #         correct += 1
    #         prediction_info = {
    #             "question": question,
    #             "answer": correct_answer,
    #             "context": context 
    #         }
    #         correct_predictions.append(prediction_info)
            
    # print(correct_predictions)
            
            
    # print(
    #     f'Accuracy: {100 * correct / (correct + incorrect)}%')
    # print(f'\tCorrect answers: {correct}')
    # print(f'\tInorrect answers: {incorrect}')

    # retriever = None
    # if args.retriever == "IR-ES":
    #     retriever = IR_ES(corpus=args.corpus)
    # elif args.retriever == "IR-CUSTOM":
    #     retriever = IR_Custom()
    # elif retriever == "ColBERT":
    #     retriever = ColBERT()
    # elif retriever == "DPR":
    #     retriever == DPR()
    
    # reader = None
    # if args.reader == "Base_BERT":
    #     reader = BERT_reader(corpus=args.corpus)
    
    
    
# 302