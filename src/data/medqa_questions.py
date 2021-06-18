import json

from nltk.stem.snowball import SnowballStemmer


class MedQAQuestions:
    questions_train_path = "data/medqa/questions/4_options/train.jsonl"
    questions_val_path = "data/medqa/questions/4_options/dev.jsonl"
    questions_test_path = "data/medqa/questions/4_options/test.jsonl"

    questions_train_filtered_path = "data/medqa/questions/4_options/[filtered]q_train_valid.json"
    questions_val_filtered_path = "data/medqa/questions/4_options/[filtered]q_val_valid.json"
    questions_test_filtered_path = "data/medqa/questions/4_options/[filtered]q_test_valid.json"

    stemmer = SnowballStemmer(language='english')

    def __init__(self, filter, stemming):
        self.stemming = stemming

        if not filter:
            print("*** Loading full sets of questions ... ***")
            self.questions_train = self.load_questions(
                questions_path=self.questions_train_path)
            self.questions_val = self.load_questions(
                questions_path=self.questions_val_path)
            self.questions_test = self.load_questions(
                questions_path=self.questions_test_path)
        else:
            print("*** Loading filtered sets of questions ... ***")
            self.questions_train = self.load_filtered_questions(
                questions_path=self.questions_train_filtered_path)
            self.questions_val = self.load_filtered_questions(
                questions_path=self.questions_val_filtered_path)
            self.questions_test = self.load_filtered_questions(
                questions_path=self.questions_test_filtered_path)

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

    def load_filtered_questions(self, questions_path):
        questions = {}

        with open(questions_path, 'r') as file:
            content = json.load(file)
            for idx, question_data in enumerate(content):
                questions[f"q{idx}"] = question_data
        return questions
