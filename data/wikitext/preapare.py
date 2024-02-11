import os
import requests
import tiktoken
import numpy as np


def read_raw_file(file_name):
    # Get the absolute path to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the file
    file_path = os.path.join(script_dir, '..', '..', 'wikitext-2-raw', file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


train_data = read_raw_file('wiki.train.raw')
test_data = read_raw_file('wiki.test.raw')
val_data = read_raw_file('wiki.valid.raw')
    
n = len(train_data)+len(test_data)


# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train has 2,417,786 tokens
# val has 249,887 tokens
