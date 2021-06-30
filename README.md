# Dense index search

This repository contains the codebase for the medical-domain multiple-choice question answering problem. The project has been conducted for the Master Thesis of Gustaw Żyngiel for obtaining the MSc in Computer Science and Engineering degree.

## Data

The data used in this project can be found in the [MedQA repository](https://github.com/jind11/MedQA). To make the code runnable, the medical textbooks as well as the USMLE questions need to be inserted in the `data` directory in the corresponding subfolders (see `notes.md` files within the `data` directory).

## Implemented solutions

The implemented solutions are:

- IR-ES e2e: retrieval using the Elasticsearch index and choosing the answer based on BM25 scores
- IR-ES + BERT reader: retrieval using the Elasticsearch index and choosing the answer by using BERT-based reader component
- REALM-like + BERT reader: retrieval using the REALM-like approach and choosing the answer by using BERT-based reader component
- ColBERT e2e: retrieval using the ColBERT-like workflow and choosing the answer based on the cosine similarity scores

## Running solutions

Run `python src/main.py <mode> <retriever> <reader> <optional flags>` where:

- `<mode>` can be either TRAINING or QA (the latter requires already having the saved trained model checkpoints)
- `<retriever>` available options: `IR-ES`, `ColBERT`, `REALM-like`
- `<reader>` available options: `Base-BERT` or `None` (the latter resulting in running the e2e solutions)
- `<optional flags>`:
  - `--colbert_base`: either `bio` or `base` indicating the choice of either BioClinicalBERT or BaseBERT uncased as the used BERT models (ColBERT based solution only)
  - `--batch_size`: default: 32
  - `--num_epochs`: default: 4


## Note on running the scripts

The solutions in the `solution` directory have the paths to:

- weights of the created models
- the checkpoints used for ColBERT solution
- the downloaded tensorflow REALM-embedder model

hardcoded. In order to run them, you need to:

1. Change the values of flags `create_encodings` and `create_index` to `True` in the `main.py` line 46.
2. Obtain the checkpoints for ColBERT and change the paths in the `colbert/colbert_retriever`. The generated and used checkpoints were achieved by running the `train.py` script from the [official ColBERT repository](https://github.com/stanford-futuredata/ColBERT) (ColBERT based solution only).
3. Create a directory `data/realm-tf-to-pytorch` and save there the downloaded tensorflow model from the [official checkpoint release](https://console.cloud.google.com/storage/browser/realm-data/cc_news_pretrained/embedder;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false) (REALM-like solution only).
4. Download the Elasticsearch 7.11 and create the index for stroring the created document chunks (IR-ES based methods only).

Perhaps in the nearest future the code will be more parametrized to make its usage easier.