import os
from nltk.stem.snowball import SnowballStemmer


class MedQACorpus():
    stemmer = SnowballStemmer(language='english')
    textbooks_data_dir = 'data/medqa/textbooks/'

    def __init__(self, stemming=False) -> None:
        self.stemming = stemming
        self.corpus = self.load_corpus()

    def load_corpus(self):
        def stem_content(content):
            tokens = [self.stemmer.stem(x) for x in content.split()]
            return ' '.join(tokens)
        corpus = {}

        for textbook_name in os.listdir(self.textbooks_data_dir):
            textbook_path = self.textbooks_data_dir + '/' + textbook_name
            with open(textbook_path, 'r') as textbook_file:
                textbook_content = textbook_file.read()
                if self.stemming:
                    textbook_content = stem_content(textbook_content)
                corpus[textbook_name] = textbook_content
        return corpus
