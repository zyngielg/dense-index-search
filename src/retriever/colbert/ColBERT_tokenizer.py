import torch

from transformers import BertTokenizerFast


class ColbertTokenizer():
    def __init__(self, bert_name, query_maxlen, doc_maxlen):
        self.query_tokenizer = BertTokenizerFast.from_pretrained(bert_name)
        self.doc_tokenizer = BertTokenizerFast.from_pretrained(bert_name)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen

        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tokenizer.convert_tokens_to_ids(
            '[unused0]')
        self.D_marker_token, self.D_marker_token_id = '[D]', self.tokenizer.convert_tokens_to_ids(
            '[unused1]')

        self.cls_token, self.cls_token_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.sep_token, self.sep_token_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id

        assert self.Q_marker_token_id == 1
        assert self.D_marker_token_id == 2
        assert self.mask_token_id == 103

    def tokenize_documents(self, documents):
        tokens = [self.doc_tokenizer.tokenize(
            x, add_special_tokens=False) for x in documents]

        prefix, suffix = [self.cls_token,
                          self.D_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix for lst in tokens]

        return tokens

    def encode_documents(self, documents):
        ids = self.doc_tokenizer(documents, add_special_tokens=False)[
            'input_ids']

        prefix, suffix = [self.cls_token_id,
                          self.D_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix for lst in ids]

        return ids

    def tensorize_documents(self, documents):
        # add placehold for the [D] marker
        documents = ['. ' + x for x in documents]

        obj = self.doc_tokenizer(documents, padding='longest', truncation='longest_first',
                                 return_tensors='pt', max_length=self.doc_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        return ids, mask

    def tokenize_queries(self, queries):
        tokens = [self.query_tokenizer.tokenize(
            x, add_special_tokens=False) for x in queries]

        prefix, suffix = [self.cls_token,
                          self.Q_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix + [self.mask_token] *
                  (self.query_maxlen - (len(lst)+3)) for lst in tokens]

        return tokens

    def encode_queries(self, queries, add_special_tokens=False):
        ids = self.query_tokenizer(queries, add_special_tokens=False)['input_ids']

        prefix, suffix = [self.cls_token_id,
                          self.Q_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix + [self.mask_token_id] *
               (self.query_maxlen - (len(lst)+3)) for lst in ids]

        return ids

    def tensorize_queries(self, queries):
        # add placehold for the [Q] marker
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='max_length', truncation=True,
                       return_tensors='pt', max_length=self.query_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id

        return ids, mask