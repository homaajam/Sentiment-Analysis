import torch

def evaluate(model, dataloader, device):
  model.eval()
  correct= 0
  total= 0

  with torch.no_grad():
    for x, y in dataloader:
      x, y= x.to(device), y.to(device)

      outputs= model(x).squeeze()

      preds= (outputs> 0.5).int()
      correct+= (preds == y).sum().item()

      total+= y.size(0)

      return correct / total