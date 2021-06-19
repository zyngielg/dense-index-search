from transformers import BertTokenizerFast


class ColbertTokenizer():
    def __init__(self, bert_name, query_maxlen, doc_maxlen):
        self.query_tokenizer = BertTokenizerFast.from_pretrained(bert_name)
        self.doc_tokenizer = BertTokenizerFast.from_pretrained(bert_name)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen

        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.query_tokenizer.convert_tokens_to_ids(
            '[unused1]')
        self.D_marker_token, self.D_marker_token_id = '[D]', self.doc_tokenizer.convert_tokens_to_ids(
            '[unused2]')

        self.cls_token, self.cls_token_id = self.doc_tokenizer.cls_token, self.doc_tokenizer.cls_token_id
        self.sep_token, self.sep_token_id = self.doc_tokenizer.sep_token, self.doc_tokenizer.sep_token_id
        self.mask_token, self.mask_token_id = self.doc_tokenizer.mask_token, self.doc_tokenizer.mask_token_id

        assert self.Q_marker_token_id == 1
        assert self.D_marker_token_id == 2
        assert self.mask_token_id == 103

    def tensorize_documents(self, documents):
        # add placehold for the [D] marker
        documents = ['. ' + x for x in documents]

        obj = self.doc_tokenizer(documents, padding='longest', truncation='longest_first',
                                 return_tensors='pt', max_length=self.doc_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        return ids, mask

    def tensorize_queries(self, queries):
        # add placehold for the [Q] marker
        batch_text = ['. ' + x for x in queries]

        obj = self.query_tokenizer(batch_text, padding='max_length', truncation=True,
                                   return_tensors='pt', max_length=self.query_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id

        return ids, mask
