from torch.utils.data import DataLoader, Dataset, SequentialSampler, TensorDataset, dataset
from tqdm import tqdm


class MedQADataset(TensorDataset):
    def __init__(self, all_possible_queries, answers, answers_idx):
        self.all_possible_queries = all_possible_queries
        self.answers = answers
        self.answers_idx = answers_idx

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.all_possible_queries)

    def __getitem__(self, index):
        'Generates one sample of data'
        queries = self.all_possible_queries[index]
        correct_answer = self.answers[index]
        correct_answer_idx = self.answers_idx[index]
        return queries, correct_answer, correct_answer_idx


class QuestionsDataset(Dataset):
    def __init__(self, questions, answers, answers_idx, options, metamap_phrases) -> None:
        self.questions = questions
        self.answers = answers
        self.answers_idx = answers_idx
        self.options = options
        self.metamap_phrases = metamap_phrases

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        answer = self.answers[index]
        answer_idx = self.answers_idx[index]
        options = self.options[index]
        metamap_phrase = self.metamap_phrases[index]

        return question, answer, answer_idx, options, metamap_phrase


def create_medqa_data_loader(input_queries, input_answers, input_answers_idx, batch_size):
    dataset = MedQADataset(all_possible_queries=input_queries,
                           answers=input_answers,
                           answers_idx=input_answers_idx)

    return DataLoader(dataset=dataset,
                      sampler=SequentialSampler(dataset),
                      batch_size=batch_size)


def create_questions_data_loader(questions, tokenizer, batch_size):
    questions_text = [x['question'] for x in questions.values()]
    answers = [x['answer'] for x in questions.values()]
    answers_idx = [ord(x['answer_idx']) - 65 for x in questions.values()]
    options = [list(x['options'].values()) for x in questions.values()]
    metamap_phrases = [x['metamap_phrases'] for x in questions.values()]

    questions_text_max_len = len(max(questions_text, key=len))
    answers_max_len = len(max(answers, key=len))
    metamap_phrases_max_len = len(max(metamap_phrases, key=len))
    # TODO: metamap_phrases issue needs to be resolved

    # questions_text = [x + ' ' * (questions_text_max_len - len(x))
    #                   for x in questions_text]
    # answers = [x + ' ' * (answers_max_len - len(x)) for x in answers]
    metamap_phrases = [
        x + [''] * (metamap_phrases_max_len - len(x)) for x in metamap_phrases]

    dataset = QuestionsDataset(questions=questions_text, answers=answers,
                               answers_idx=answers_idx, options=options, metamap_phrases=metamap_phrases)

    return DataLoader(dataset=dataset,
                      sampler=SequentialSampler(dataset),
                      batch_size=batch_size)
