import json

from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm


class MedQAQuestions:
    questions_train_path = "data/medqa/questions/4_options/train.jsonl"
    questions_dev_path = "data/medqa/questions/4_options/dev.jsonl"
    questions_test_path = "data/medqa/questions/4_options/test.jsonl"
    stemmer = SnowballStemmer(language='english')

    def __init__(self, stemming):
        self.stemming = stemming
        self.questions_train = self.load_questions(
            questions_path=self.questions_train_path)
        self.questions_dev = self.load_questions(
            questions_path=self.questions_dev_path)
        self.questions_test = self.load_questions(
            questions_path=self.questions_test_path)

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
                    for i in range(len(question['metamap_phrases'])):
                        question['metamap_phrases'][i] = stem_content(
                            question['metamap_phrases'][i])

                questions[f"q{idx}"] = question
        return questions
