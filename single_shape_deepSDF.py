import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import custom_datasets as cds

TRAIN_DATA = cds.sdSphere(-1.5, 100000)
TEST_DATA = cds.sdSphere(-1.5, 100)

TOL = 1.5
class ssdSDF(nn.Module):
    def __init__(self, layer_size, layer_number):
        super(ssdSDF, self).__init__()

        self.model = nn.Sequential()
        self.model.append(nn.Linear(3,layer_size))
        self.model.append(nn.ReLU())
        for _ in range(layer_number):
            self.model.append(nn.Linear(layer_size, layer_size))
            self.model.append(nn.ReLU())

        self.model.append(nn.Linear(layer_size,1))

        

    def forward(self, x):
        return torch.squeeze(self.model(x))

def ssd_Loss(x,y):
    delta = torch.full_like(x, TOL)
    clamp_x = torch.minimum(delta, torch.maximum(-delta, x))
    clamp_y = torch.minimum(delta, torch.maximum(-delta, y))

    l1_loss = nn.L1Loss(reduction='sum')
    return l1_loss(clamp_x, clamp_y)
    
def train_ssdSDF(dataloader, model, loss_fn):
    learning_rate = 0.0001
    batch_size = 64


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0


    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():

    dl = torch.utils.data.DataLoader(TRAIN_DATA, batch_size=64)
    dl_test = torch.utils.data.DataLoader(TEST_DATA, batch_size=64)


    model = ssdSDF(6,3)

    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t+0}\n-------------------------------")
        train_ssdSDF(dl, model, ssd_Loss)
        test_loop(dl_test, model, ssd_Loss)
    print("Done!")
    return model

if __name__ == "__main__":
    main()
