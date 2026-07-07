import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self, vocab_size, embed_dem, hidden_dem):
    super().__init__()

    self.embedding= nn.embeding(vocab_size, embed_dem)

    self.biLSTM= nn.LSTM(
      embed_dem,
      hidden_dem,
      batch_first=True,
      bidirectional=True
    )

    self.gru= nn.GRU(
      hidden_dem *2,
      hidden_dem,
      batch_first=True
    )

    self.dropout= nn.Dropout(0.3)

    self.full_conect= nn.Linear(hidden_dem, 1)

    self.sigmoid= nn.Sigmoid()

  def forward()