import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
  def __init__(self, texts, labels, vocab, max_len=100):
    self.texts= texts
    self.labels= labels
    self.vocab= vocab
    self.max_len= max_len
  
  def padding(self, seq):
    if len(seq) < self.max_len:
      seq += [0] * (self.max_len - len(seq))
    else:
      seq= seq[:self.max_len]
    return seq
  
  def __len__(self):
    return len(self.texts)
  
  def __getitem__(self, idx):
    tokens= self.texts[idx]
    encoded= self.vocab.encode(tokens)
    padded= self.padding(encoded)

    return torch.tensor(padded), torch.tensor(self.labels[idx])
  
