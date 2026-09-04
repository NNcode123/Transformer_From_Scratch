from pathlib import Path

import torch
from transformers import AutoTokenizer

from Model.transformer import Transformer


def run_tests():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    with open(data_dir / "english.txt", "r", encoding="utf8") as source_file:
        source_text = source_file.readline().strip()

    encoded_source = tokenizer(source_text, return_tensors="pt")
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        num_blocks=1,
        embed_dim=128,
        num_heads=4,
    )
    model.eval()

    with torch.no_grad():
        generated_ids = model.generate(
            encoded_source["input_ids"],
            encoded_source["attention_mask"],
        )

    print("Source:", source_text)
    print("Generated:", tokenizer.decode(generated_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    run_tests()
