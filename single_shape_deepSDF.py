import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TOL = 0.1
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
        return self.model(x)

def mod_L1_loss(x,y):
    l1=nn.L1Loss(reduction='sum')
    return l1(clamp(x,TOL), clamp(y,TOL))

def clamp(x, delta):
    return torch.minimum(torch.full_like(x, delta), torch.maximum(torch.full_like(x,-delta), x))
    
def train_ssdSDF(dataloader, model, loss_fn):
    learning_rate = 1e-3
    batch_size = 64


    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":

    class sdSphere(torch.utils.data.Dataset):
        def __init__(self, radius=0.5, samples = 64000):
            self.samples = samples
            self.coords = np.asarray(np.random.default_rng().uniform(-1.0,1.0,(self.samples,3),), dtype=np.float32)
            self.sdf = np.linalg.norm(self.coords, axis=1) - radius
        def __len__(self):
            return self.samples
        def __getitem__(self, index):
            return self.coords[index], self.sdf[index]




    dl = torch.utils.data.DataLoader(sdSphere(), batch_size=64)
    dl_test = torch.utils.data.DataLoader(sdSphere(samples=100), batch_size=64)


    model = ssdSDF(64,5)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_ssdSDF(dl,model, nn.MSELoss)
        test_loop(dl_test, model, nn.MSELoss)
    print("Done!")