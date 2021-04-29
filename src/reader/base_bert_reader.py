import torch
from reader.reader import Reader
from models.CustomBERT import CustomBERT
from transformers import AutoTokenizer

class Base_BERT_Reader(Reader):
    def __init__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using {} device".format(device))
        
        self.model = CustomBERT().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        
        
    def choose_answer(self, query, context, question_data):
        return "Nice"

    def create_context(self):
        return super().create_context()
    
    # def create_tokenized_input(question_data, documents_collection_dict: dict):
    #     input_queries = []
    #     input_answers = []
    #     input_answers_idx = []

    #     for question_id, question_data in tqdm(questions_dict.items()):
    #         question = question_data['question']
    #         metamap_phrases = question_data['metamap_phrases']
    #         queries = []
    #         for option in question_data['options'].values():
    #         qa = ' '.join(metamap_phrases) + ' ' + option
    #         retrieved_documents = get_context(question_id=question_id,
    #                                           option=option,
    #                                           documents_collection=documents_collection_dict)
    #         context = ' '.join(retrieved_documents)
    #         query = tokenizer(context, qa,
    #                           add_special_tokens=True,
    #                           max_length=512,
    #                           padding='max_length',
    #                           truncation=True,
    #                           return_tensors="pt"
    #                           )
    #         query_input_ids = query["input_ids"].flatten()
    #         query_token_type_ids = query["token_type_ids"].flatten()
    #         query_attention_mask = query["attention_mask"].flatten()

    #         queries.append({
    #             "input_ids": query_input_ids,
    #             "token_type_ids": query_token_type_ids,
    #             "attention_mask": query_attention_mask
    #         })
    #         # break
    #         # dev_dataset_input.append({
    #         #       "correct_answer": question_data["answer"],
    #         #       "correct_answer_idx": letter_answer_to_index(question_data['answer_idx']),
    #         #       "queries": queries
    #         #   })
    #         input_queries.append(queries)
    #         input_answers.append(question_data["answer"])

    #         input_answers_idx.append(
    #             letter_answer_to_index(question_data['answer_idx']))
    #     return input_queries, input_answers, input_answers_idx
