import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_dim):
    super().__init__()

    self.embedding= nn.Embedding(vocab_size, embed_dim)

    self.biLSTM= nn.LSTM(
      embed_dim,
      hidden_dim,
      batch_first=True,
      bidirectional=True
    )

    self.gru= nn.GRU(
      hidden_dim *2,
      hidden_dim,
      batch_first=True
    )

    self.dropout= nn.Dropout(0.3)

    self.fc= nn.Linear(hidden_dim, 1)

    self.sigmoid= nn.Sigmoid()

  def forward(self, x):
    x= self.embedding(x)

    lstm_out, _ = self.biLSTM(x)
    gru_out, _ = self.gru(lstm_out)

    x= gru_out[:,-1,:]
    x= self.dropout(x)
    x= self.fc(x)

    return self.sigmoid(x)
  
