import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset , DataLoader
import pandas as pd




class Spam_Dataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length = None, pad_token_id = 50256):
        super().__init__()

        self.df = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.df["Text"]]
        

        if max_length is None:
            self.max_length = self.Longest_encoded_length()
        else:
            self.max_length = max_length
        ## Truncating the sequences if they are longer than max_length
        self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
        self.encoded_texts = [encoded_text + [pad_token_id]*(self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.df.iloc[index]["Label"]
        return (torch.tensor(encoded, dtype = torch.long),
                torch.tensor(label, dtype = torch.long))

    def __len__(self):
        return len(self.df)

    def Longest_encoded_length(self):
        max_length =0
        for text in self.encoded_texts:
            encoded_len = len(text)
            if encoded_len> max_length :
                max_length = encoded_len

        return max_length


tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = Spam_Dataset(csv_file = "../code/sms_spam_collection/train.csv", tokenizer = tokenizer, max_length = None)
val_dataset = Spam_Dataset(csv_file = "../code/sms_spam_collection/val.csv", tokenizer = tokenizer, max_length = None)
test_dataset = Spam_Dataset(csv_file = "../code/sms_spam_collection/test.csv", tokenizer = tokenizer, max_length = None)


num_workers =0
batch_size =8
torch.manual_seed(123)
train_loader = DataLoader(train_dataset,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = num_workers,
                          drop_last = True)

val_loader = DataLoader(val_dataset,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = num_workers,
                          drop_last = True)

test_loader = DataLoader(test_dataset,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = num_workers,
                          drop_last = True)

# small_data = next(iter(train_loader))
# print(small_data.shape)

for input_batch, label_batch in train_loader:
    pass
print(f"Input batch shape :{input_batch}")
print(f"Label batch shape :{label_batch}")