import torch

def train(model, dataloader, optimizer, criterion, device):
  model.train()
  total_loss= 0
  
  for x, y in dataloader:
    x, y= x.to(device), y.to(device).float()

    optimizer.zero_grad()

    output= model(x).squeeze()

    loss= criterion(output, y)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

  return total_loss/len(dataloader)

