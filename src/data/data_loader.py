from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
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


def create_data_loader(input_queries, input_answers, input_answers_idx, batch_size):
    dataset = MedQADataset(all_possible_queries=input_queries,
                           answers=input_answers,
                           answers_idx=input_answers_idx)

    return DataLoader(dataset=dataset,
                      sampler=SequentialSampler(dataset),
                      batch_size=batch_size)


