import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TokenDataset(Dataset):
    def __init__(self, english_texts, german_texts, tokenizer=None):
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained("t5-small")
        self.english_texts = english_texts
        self.german_texts = german_texts

        self.src_ids = self._tokenize(self.english_texts)["input_ids"]
        self.src_attention_mask = self._tokenize(self.english_texts)["attention_mask"]

        self.tgt_ids = self._tokenize(self.german_texts)["input_ids"]
        self.tgt_attention_mask = self._tokenize(self.german_texts)["attention_mask"]

    def _tokenize(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    def __getitem__(self, idx: int):
        return {
            "src_ids": self.src_ids[idx],
            "src_att_mask": self.src_attention_mask[idx],
            "tgt_ids": self.tgt_ids[idx],
            "tgt_mask": self.tgt_attention_mask[idx],
        }

    def __len__(self):
        return len(self.english_texts)


class TokenLoader:
    def __init__(self, dataset, batch_size=2, shuffle=True, num_workers=0):
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __iter__(self):
        return iter(self.loader)


def main():
    english_sentences = ["Hello world", "How are you", "I like Python"]
    german_sentences = ["Hallo Welt", "Wie geht es dir", "Ich mag Python"]

    dataset = TokenDataset(english_sentences, german_sentences)
    loader = TokenLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    batch = next(iter(loader))
    for key, value in batch.items():
        print(key, value.shape)


if __name__ == "__main__":
    main()
