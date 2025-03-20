import numpy as np
import torch
import torch.nn as nn
import custom_datasets as cds
import rayMarch_class


class msdSDF(nn.Module):
    def __init__(self, layer_size, layer_number, latent_size):
        super(msdSDF, self).__init__()

        self.model = nn.Sequential()
        self.model.append(nn.Linear(3+latent_size, layer_size))
        self.model.append(nn.ReLU())
        for _ in range(layer_number):
            self.model.append(nn.Linear(layer_size, layer_size))
            self.model.append(nn.ReLU())
        self.model.append(nn.Linear(layer_size, 1))

    def forward(self, x):
        return torch.squeeze(self.model(x))

    def train(self, train_data, loss_fn, learning_rate):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        size = len(train_data.dataset)
        self.model.train()
        loss_log = []
        current = 0
        for batch, (X, y) in enumerate(train_data):
            current += len(X)
            pred = torch.squeeze(self.model(X))
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_log.append(loss.item())
            if batch % 100 == 0:
                loss = loss.item()
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

        return np.array(loss_log)

    def test(self, test_data, loss_fn):
        self.model.eval()
        num_batches = len(test_data)
        test_loss = 0

        with torch.no_grad():
            for X, y in test_data:
                pred = torch.squeeze(self.model(X))
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f"Test Error: \n Avg loss: {test_loss:>8f}\n")


if __name__ == "__main__":
    sdf_func = [cds.box_sdf(np.array([0.5, 0.5, 0.5])).get_sdf, cds.sphere_sdf(0.5).get_sdf]
    training_set = torch.utils.data.DataLoader(
        cds.sdMixed(sdf_func, 10000), batch_size=64)
    testing_set = torch.utils.data.DataLoader(
        cds.sdMixed(sdf_func, 100), batch_size=64)

    inst = msdSDF(64, 5, 1)

    for i in range(5):
        inst.train(training_set, nn.MSELoss(), 0.001)
        inst.test(testing_set, nn.MSELoss())

    renderer = rayMarch_class.RAY_MARCH_RENDERER(inst, 0)
    renderer_2 = rayMarch_class.RAY_MARCH_RENDERER(inst, 1)

    #renderer.main_loop()
    #renderer_2.main_loop()
