import torch
from nltk.stem.snowball import SnowballStemmer
from reader.custom_bert import CustomBERT
from utils.data_reader import preprocess_questions, read_questions_data


snowball_stemmer = SnowballStemmer(language='english')
lr = 1e-5
criterion = torch.nn.CrossEntropyLoss()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))
model = CustomBERT().to(device)

questions_data_dev, questions_data_train, questions_data_test = read_questions_data()
preprocess_questions(questions=questions_data_dev[:160], stemmer=snowball_stemmer, remove_punctuation=False)
# preprocess_questions(questions=questions_data_train[:160], stemmer=snowball_stemmer, remove_punctuation=False)

dev_questions = [x['question'] for x in questions_data_dev]
dev_answers = [x['answer'] for x in questions_data_dev]


def train_model():
    for epoch in range(3):
        print(f"Epoch {epoch}:")
        
