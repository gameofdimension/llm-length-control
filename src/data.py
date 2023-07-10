import nltk
from datasets import load_dataset, DatasetDict


def train():
    train_urls = [
        f'https://worksheets.codalab.org/rest/bundles/0x8b65ebfe46674fbc83fc6df60da32f1b/contents/blob/openwebtext_wordlength/openwebtext_wordlength_train.json_{i}'
        for i in range(10)]
    pubmed_dataset_streamed = load_dataset(
        "json", data_files=train_urls, split="train"
    )
    return pubmed_dataset_streamed


def validate():
    train_urls = [
        f'https://worksheets.codalab.org/rest/bundles/0x8b65ebfe46674fbc83fc6df60da32f1b/contents/blob/openwebtext_wordlength/openwebtext_wordlength_val.json_{i}'
        for i in range(10)]
    pubmed_dataset_streamed = load_dataset(
        "json", data_files=train_urls, split="test"
    )
    return pubmed_dataset_streamed


def split(text):
    text = text.replace('<len> ', '|$$$$|<len> ')
    return [v.strip() for v in text.split("|$$$$|") if v.strip() != '']


def explode(batch):
    longtext = batch['text']
    lst = []
    for text in longtext:
        lst.extend(split(text))
    value = {'sentence': lst}
    return value


def make_data():
    train_data = train()
    test_data = validate()

    test_data = test_data.map(explode, remove_columns=["text"], batched=True)
    train_data = train_data.map(explode, remove_columns=["text"], batched=True)

    split_datasets = train_data.train_test_split(train_size=0.9, seed=20)
    raw_datasets = DatasetDict(
        {
            "train": split_datasets['train'].shuffle().select(range(4_000_000)),  # .shuffle().select(range(50000)),
            "valid": split_datasets['test'].shuffle().select(range(450_000)),  # .shuffle().select(range(500))
            "test": test_data
        }
    )
    raw_datasets.push_to_hub("openwebtext-wordlength")
