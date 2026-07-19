import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from source.preprocess import clean_text, tokenize
from utils.vocab import Vocab
from source.dataset import TextDataset
from source.model import Model
from source.train import train
from source.evaluate import evaluate


df= pd.read_csv("data/imdb.csv")
texts= [tokenize(clean_text(t)) for t in df["review"]]
labels= df["sentiment"].map({"positive":1, "negative":0}).tolist()

vocab= Vocab()
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2
)
vocab.build_vocab(train_texts)


train_dataset = TextDataset(train_texts, train_labels, vocab)
val_dataset = TextDataset(val_texts, val_labels, vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= Model(len(vocab.word2idx), 100, 64).to(device)
optimizer= torch.optim.Adam(model.parameters(), lr=0.001)
criterion= torch.nn.BCELoss()

for epoch in range(5):
  loss= train(model, train_loader, optimizer, criterion, device)
  train_acc= evaluate(model, train_loader, device)
  val_acc= evaluate(model, val_loader, device)
  print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Train_Acc: {train_acc:.4f} | Vaal_Acc: {val_acc:.4f}")


