This repo has two knowledge distillation.
    
    Knowledge-Distillation-Small: 
        Run the knowledge_distillation.ipynb file, it is able to run on a gpu google colab environment.


    Knowledge-Distillation-Large: 
        1. from datasets import load_dataset
            dataset = load_dataset("openwebtext")

            download the openwebtext from https://huggingface.co/datasets/openwebtext
        2. I used https://cloud.lambdalabs.com/ 1X A10 (24 GB PCle) , 30 vCPUs, 200 GiB RAM, 1.4 TiB SSD for training
        3. python distillation/scripts/binarized_data.py \
            --file_path data/urlsf_subset00-1_data \
            --tokenizer_type gpt2 \
            --tokenizer_name gpt2 \
            --dump_file data/binarized_text_valid
            Change the file name and path accordingly. This step is to transfer the text data to binarized data,
            tokenize the data and convert each token in an index in GPT2's vocabulary.
        4. python distillation/scripts/token_counts.py \
            --data_file data/binarized_text_valid.gpt2.pickle \
            --token_counts_dump data/token_counts.binarized_text_valid.gpt2.pickle \
            --vocab_size 50257
            Change the file name and path accordingly. This step is to count the occurrences of each tokens in the data.
        5. python distillation/train.py --student_type gpt2 
            --student_config distillation/training_configs/distilgpt2.json --teacher_type gpt2 --teacher_name gpt2 
            --alpha_ce 5.0 --alpha_mlm 0.0 --alpha_cos 1.0 --alpha_clm 2.0 --freeze_pos_embs 
            --dump_path serialization_dir/my_first_training --data_file data/binarized_text_train.gpt2.pickle 
            --token_counts data/token_counts.binarized_text_train.gpt2.pickle --force 
            Change the file name and path accordingly. This step is to train the model.
                        