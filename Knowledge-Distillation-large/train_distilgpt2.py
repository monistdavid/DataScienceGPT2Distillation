from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM

dataset = load_dataset("openwebtext")
# teacher_tokenizer = AutoTokenizer.from_pretrained("gpt2")
# teacher_model = AutoModelForCausalLM.from_pretrained("gpt2")
#
# tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')
#
#
# def tokenize_function(example):
#     return tokenizer(example['text'])
#
# tokenized_dataset = dataset.map(tokenize_function, batched=True)
#
# from transformers import DataCollatorForLanguageModeling
#
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer, mlm=False, return_tensors="pt",
# )
#
#
# from transformers import TrainingArguments
#
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=64,
#     eval_steps=500,
#     save_steps=1000,
#     warmup_steps=500,
#     logging_dir='./logs',
#     logging_steps=1000,
#     learning_rate=1e-4,
#     weight_decay=0.01
# )