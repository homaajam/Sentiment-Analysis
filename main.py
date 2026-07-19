import pandas as pd
import torch
from torch.utils.data import DataLoader
from source.preprocess import clean_text, tokenize
from utils.vocab import Vocab
from source.dataset import TextDataset


df= pd.read_csv("data/imdb.csv")
texts= [tokenize(clean_text(t)) for t in df["review"]]
labels= df["sentiment"].map({"positive":1, "negative":0}).tolist()

vocab= Vocab()
vocab.build_vocab(texts)

dataset= TextDataset(texts, labels, vocab)
loader= DataLoader(dataset, batch_size=32, shuffel=True)

