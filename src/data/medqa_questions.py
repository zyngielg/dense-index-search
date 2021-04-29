import json

from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm


class MedQAQuestions:
    questions_train_path = "data/medqa/questions/train.jsonl"
    questions_dev_path = "data/medqa/questions/4_options/dev.jsonl"
    questions_test_path = "data/medqa/questions/test.jsonl"
    stemmer = SnowballStemmer(language='english')
    
    def __init__(self, stemming):
        self.stemming = stemming
        self.questions_train = self.load_questions(questions_path=self.questions_train_path)
        self.questions_dev = self.load_questions(questions_path=self.questions_dev_path)
        self.questions_test = self.load_questions(questions_path=self.questions_test_path)

    
    def load_questions(self, questions_path):
        def stem_content(content):
            tokens = [self.stemmer.stem(x) for x in content.split()]
            return ' '.join(tokens)
        
        questions = {}

        with open(questions_path, 'r') as file:
            for idx, line in enumerate(file):
                question = json.loads(line)
                if self.stemming:
                    question['question'] = stem_content(question['question'])
                    question['answer'] = stem_content(question['answer'])
                    for option, value in question['options'].items():
                        question['options'][option] = stem_content(value)

                questions[f"q{idx}"] = question
            
        return questions
    

# def ir_es(questions, no_documents_to_retrieve, index_name):
#     correct_answer = 0
#     incorrect_answer = 0

#     for question_data in tqdm(questions):
#         question = question_data['question']
#         answer = question_data['answer']

#         final_answer = None
#         final_score = 0

#         for option, option_answer in question_data['options'].items():
#             # query = ' '.join(question_data['metamap_phrases']) + " " + option_answer
#             query = question + " " + option_answer
#             top_documents = search_documents(
#                 query, no_documents_to_retrieve, index_name)
#             if top_documents != []:
#                 score = ir_es_score(top_documents)
#                 if final_score < score:
#                     final_answer = option_answer
#                     final_score = score

#         correct = False
#         if final_answer == answer:
#             correct_answer += 1
#             correct = True
#         else:
#             incorrect_answer += 1

#     print(
#         f'Accuracy: {100 * correct_answer / (correct_answer + incorrect_answer)}%')
#     print(f'\tCorrect answers: {correct_answer}')
#     print(f'\tInorrect answers: {incorrect_answer}')


# def calculate_average_query_doc_len(questions_medqa, index_name):
#     no_all_queries_tokens = []
#     no_all_context_tokens = []
#     for question_data in tqdm(questions_medqa):
#         for answer_option in question_data['options'].values():
#             query_tokens = question_data['metamap_phrases']
#             query_tokens.append(answer_option)
#             query = ' '.join(query_tokens)
            
#             context_documents_info = search_documents(
#                 query, no_documents_to_retrieve, index_name)
#             context_documents = [x['evidence']['content'] for x in context_documents_info]
            
#             no_all_queries_tokens.append(len(query_tokens))
#             no_all_context_tokens.append([len(word_tokenize(x)) for x in context_documents])
    
#     return np.mean(no_all_queries_tokens), np.mean(no_all_context_tokens)
            

# def ir_es_custom(questions_medqa, no_documents_to_retrieve, index_name):
#     correct_answer = 0
#     incorrect_answer = 0

#     # create list of (question, query, context)
#     print('Calculating average query and context lengths...')
#     avg_query_len, avg_context_len = calculate_average_query_doc_len(
#         questions_medqa, index_name)
#     print('... average query and context lenghts calculated ')
    
#     print('Retrieving answers to questions...')
#     for question_data in tqdm(questions_medqa):
#         answer = question_data['answer']
#         answer_option = question_data['answer_idx']
        
#         final_answer = None
#         final_score = 0

#         for option, option_answer in question_data['options'].items():
#             query = ' '.join(
#                 question_data['metamap_phrases']) + " " + option_answer
#             top_documents = search_documents(
#                 query, no_documents_to_retrieve, index_name)

#             if top_documents != []:
#                 documents_content = [x['evidence']['content'] for x in top_documents]
                
#                 score = ir_es_custom_score(documents_content, query, avg_query_len, avg_context_len)
#                 if final_score < score:
#                     final_answer = option_answer
#                     final_score = score

#         correct = False
#         if final_answer == answer:
#             correct_answer += 1
#             correct = True
#         else:
#             incorrect_answer += 1

#     print(
#         f'Accuracy: {100 * correct_answer / (correct_answer + incorrect_answer)}%')
#     print(f'\tCorrect answers: {correct_answer}')
#     print(f'\tInorrect answers: {incorrect_answer}')
#     print('... answers to questions retrieved')

# questions_dev = load_questions(questions_dev_medqa_path)
# questions_train = load_questions(questions_train_medqa_path)

# questions_dev_stemmed = copy.deepcopy(questions_dev)
# stem_questions(questions_dev_stemmed, snowball_stemmer)

# questions_train_stemmed = stem_questions(copy.deepcopy(questions_train), snowball_stemmer)
# stem_questions(questions_train_stemmed, snowball_stemmer)

# print('********** IR_ES **********')
# print('\tDev:')
# ir_es(questions=questions_dev, no_documents_to_retrieve=no_documents_to_retrieve, index_name=Indexes.Unprocessed_sentences_shards_1.value)
# print('\tTrain:')
# ir_es(questions=questions_train, no_documents_to_retrieve=no_documents_to_retrieve, index_name=Indexes.Unprocessed_sentences_shards_1.value)

# print('********** IR_ES **********')
# print('\tDev:')
# ir_es_custom(questions_medqa=questions_dev_stemmed, no_documents_to_retrieve=no_documents_to_retrieve,
#              index_name=Indexes.Stemmed_sentences_shards_1.value)
# print('\tTrain:')
# ir_es_custom(questions_medqa=questions_train_stemmed, no_documents_to_retrieve=no_documents_to_retrieve,
#              index_name=Indexes.Stemmed_sentences_shards_1.value)