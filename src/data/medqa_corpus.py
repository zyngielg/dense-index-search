import os
from nltk.stem.snowball import SnowballStemmer
from nltk import sent_tokenize, word_tokenize
from itertools import filterfalse


class MedQACorpus():
    stemmer = SnowballStemmer(language='english')
    textbooks_dir = 'data/medqa/textbooks/'
    filtered_textbooks_dir = 'data/medqa/textbooks_filtered/'
    filtered_textbooks_log = 'data/medqa/textbooks_filtering_log.txt'
    short_sentence_counter = 0
    long_word_sentences_counter = 0
    num_words_removed = 0

    def __init__(self, stemming=False) -> None:
        self.stemming = stemming
        self.corpus = self.load_corpus()

    def load_corpus(self):
        def stem_content(content):
            tokens = [self.stemmer.stem(x) for x in content.split()]
            return ' '.join(tokens)
        corpus = {}

        if not os.path.exists(self.filtered_textbooks_dir):
            self.filter_textbooks()

        print("******** Generating corpus ... ********")
        for book_title_txt in os.listdir(self.filtered_textbooks_dir):
            textbook_path = self.filtered_textbooks_dir + '/' + book_title_txt
            with open(textbook_path, 'r') as textbook_file:
                textbook_content = textbook_file.read()
                if self.stemming:
                    textbook_content = stem_content(textbook_content)
                corpus[book_title_txt[:-4]] = textbook_content
        print("******** ... corpus generated ********")
        return corpus

    def filter_textbooks(self):
        os.mkdir(self.filtered_textbooks_dir)
        print("******** Filtering MedQA textbooks ... ********")
        for book_title_txt in os.listdir(self.textbooks_dir):
            textbook_path = self.textbooks_dir + '/' + book_title_txt
            with open(textbook_path, 'r') as textbook_file:
                content = textbook_file.read()

                # remove new lines and tabs
                content = content.replace('\n', ' ').replace(
                    '\r', ' ').replace('\t', ' ').strip()

                # split to sentences
                sentences = sent_tokenize(content)
                num_sentences_before = len(sentences)
                num_words_before = sum([len(word_tokenize(x))
                                       for x in sentences])
                # remove
                sentences = list(filterfalse(self.__filter_short, sentences))
                sentences = list(filterfalse(
                    self.__filter_with_long_words, sentences))
                num_removed = self.short_sentence_counter + self.long_word_sentences_counter
                num_sentences_after = len(sentences)
                num_words_after = sum([len(word_tokenize(x))
                                      for x in sentences])
                assert num_sentences_before-num_sentences_after == num_removed
                assert num_words_before-num_words_after == self.num_words_removed

                self.__log_message(
                    book_title_txt[:-4], num_sentences_before, num_sentences_after, num_words_before, num_words_after)

                self.short_sentence_counter = 0
                self.long_word_sentences_counter = 0
                self.num_words_removed = 0

                content_filtered = ' '.join(sentences)
                filtered_textbook_path = f"{self.filtered_textbooks_dir}/{book_title_txt}"
                with open(filtered_textbook_path, 'w') as filtered_textbook_file:
                    filtered_textbook_file.write(content_filtered)
        print("******** ... filtering complete ********")

    def __filter_short(self, sentence):
        tokens = word_tokenize(sentence)

        res = len(tokens) <= 5
        if res:
            self.short_sentence_counter += 1
            self.num_words_removed += len(tokens)
        return res

    def __filter_with_long_words(self, sentence):
        tokens = word_tokenize(sentence)
        max_len = 35

        lenghts = [len(x) for x in tokens]
        if any(x > max_len for x in lenghts):
            self.long_word_sentences_counter += 1
            self.num_words_removed += len(tokens)
            return True
        return False

    def __log_message(self, title, num_sent_before, num_sent_after, num_words_before, num_words_after):
        log_message = f"[{title}] \n\t#sentences: \tbefore: {num_sent_before}\tafter: {num_sent_after} -> removed: {num_sent_before-num_sent_after} sentences:"
        log_message += f"{self.short_sentence_counter} which had less than 6 words and after {self.long_word_sentences_counter} which contained a word of length over 35\n"
        log_message += f"\t#words:     \tbefore: {num_words_before}\tafter: {num_words_after} -> removed: {num_words_before-num_words_after}\n"
        print(log_message)
        with open(self.filtered_textbooks_log, 'a') as log_file:
            log_file.write(log_message)
