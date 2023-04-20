from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda"
model_id = "gpt2"
# change to your model path accordingly
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

from datasets import load_dataset

# wikitext-103
# test = load_dataset("wikitext", "wikitext-103-v1", split="test")
# encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

# enwik8
# test = load_dataset("enwik8", split="train")
# encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

# lambada
# test = load_dataset("lambada", split="test")
# encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

# lambada
# test = load_dataset("common_gen ", split="test")
# encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

# PTD Penn Treebank
test = load_dataset("ptb_text_only", split="test")
encodings = tokenizer("\n\n".join(test["sentence"]), return_tensors="pt")

print(encodings)

import torch
from tqdm import tqdm

max_length = model.config.n_positions
stride = 1024
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print("perplexity:", ppl)
