import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from os import path
import huggingface as hf
from transformers import AutoTokenizer
from typing import Union


class TokenDataset(Dataset):

    def __init__(self, text: Union[Path, str], target_text: Union[Path, str], tokenizer = None):

        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained("t5-small")
    
        with open(text, "r", encoding = "utf8") as src_file:
           text_buff = src_file.readlines()

        with open(target_text, "r", encoding= "utf8") as tgt_file:
            target_text_buff = tgt_file.readlines()

        src_info, tgt_info = self.tokenizer(text = text_buff, padding = True, truncation = True, return_tensors = "pt"), self.tokenizer(text = target_text_buff, padding = True, truncation = True,  return_tensors = "pt")

        self.src_ids = src_info["input_ids"]

        self.src_attention_mask = src_info["attention_mask"]

        self.tgt_attention_mask = tgt_info["attention_mask"]

        self.tgt_ids = tgt_info["input_ids"]

        self.data  = {}


    @classmethod
    def from_json(self, json_file):
        pass



    @classmethod
    def from_txt(self, src_txt, target_txt):
        pass


    def get_tokenizer(self):
        return self.tokenizer



    def tokenize(**kwargs):
        pass





    def __getitem__(self, idx: int):
        return {
            "src_ids": self.src_ids[idx],
            "src_att_mask": self.src_attention_mask[idx],
            "tgt_ids": self.tgt_ids[idx],
            "tgt_mask": self.tgt_attention_mask[idx]
        }

    def __len__(self):
        return self.src_ids.size(0)


    




def main():
    data_dir = Path(__file__).resolve().parent
    dataset = TokenDataset(data_dir / "english.txt", data_dir / "german.txt")
    loader = DataLoader(dataset, batch_size=3, shuffle=True)
    batch = next(iter(loader))

    print("batch keys:", list(batch.keys()))

    print("src_ids shape:", tuple(batch["src_ids"].shape))


    tokenizer = dataset.get_tokenizer()

    
    print(batch["src_ids"])

    print(f"Source Text:\n\n {tokenizer.decode(batch["src_ids"])}")


    print("src_att_mask shape:", tuple(batch["src_att_mask"].shape))
    print(batch["src_att_mask"])

    print("tgt_ids shape:", tuple(batch["tgt_ids"].shape))
    print(batch["tgt_ids"])

    print(f"Target Text:\n\n {tokenizer.decode(batch["tgt_ids"])}")

    print("tgt_mask shape:", tuple(batch["tgt_mask"].shape))
    print(batch["tgt_mask"])


if __name__ == "__main__":
    main()

