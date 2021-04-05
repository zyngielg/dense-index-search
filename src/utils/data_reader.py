import json
import string
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm 

questions_dev_medqa_path = '../data/medqa/questions/metamap_extracted_phrases/dev.jsonl'
questions_train_medqa_path ='../data/medqa/questions/metamap_extracted_phrases/train.jsonl'
questions_test_medqa_path ='../data/medqa/questions/metamap_extracted_phrases/train.jsonl'

snowball_stemmer = SnowballStemmer(language='english')

def read_questions_data():
    questions_data_dev = []
    questions_data_train = []
    questions_data_test = []

    with open(questions_dev_medqa_path, 'r') as file:
        for line in file:
            questions_data_dev.append(json.loads(line))

    with open(questions_train_medqa_path, 'r') as file:
        for line in file:
            questions_data_train.append(json.loads(line))
            
    with open(questions_test_medqa_path, 'r') as file:
        for line in file:
            questions_data_test.append(json.loads(line))  
            
    return questions_data_dev, questions_data_train, questions_data_test
    

def preprocess_content(content, stemmer, remove_punctuation):
    if not stemmer and not remove_punctuation:
        return content.lower()
    if remove_punctuation:
        custom_string_punctuation = string.punctuation.replace('-','').replace('/','').replace('.','')
        punctuation = str.maketrans('', '', custom_string_punctuation)
        content = content.translate(punctuation).replace('“','').replace('’','')
    
    sentences = sent_tokenize(content.lower())
    cleaned_sentences = []
    
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        if stemmer:
            tokens = [stemmer.stem(x) for x in tokens]
        cleaned_sentences.append(' '.join(tokens))
            
    return ' '.join(cleaned_sentences)


def preprocess_questions(questions, stemmer, remove_punctuation):    
    for question in tqdm(questions):
        question['question'] = preprocess_content(question['question'], stemmer, remove_punctuation)
        for option, value in question['options'].items():
            question['options'][option] = preprocess_content(value, stemmer, remove_punctuation)        
            question['answer'] = preprocess_content(question['answer'], stemmer, remove_punctuation)
            for i, phrase in enumerate(question['metamap_phrases']):
                question['metamap_phrases'][i] = preprocess_content(phrase, stemmer, remove_punctuation)

# Example usage:
# questions_data_dev, questions_data_train, questions_data_test = read_questions_data()
# preprocess_questions(questions=questions_data_dev[:10], stemmer=snowball_stemmer, remove_punctuation=False)