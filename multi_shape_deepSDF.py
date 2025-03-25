import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import trimesh
from mesh_to_sdf import sample_sdf_near_surface as sns


# Konstants
SIGMA = 0.01
TOL = 0.1


class msdSDF(nn.Module):
    def __init__(self, layer_size: int,
                 layer_number: int,
                 latent_size: int,
                 loss_fn):

        super(msdSDF, self).__init__()

        # setup class variables
        self.loss_fn = loss_fn
        self.latent_size = latent_size

        # Setup network
        self.model = nn.Sequential()

        # Setup input layer
        self.model.append(nn.Linear(3+latent_size, layer_size))
        self.model.append(nn.ReLU())

        # Setup hidden Layers
        for _ in range(layer_number):
            self.model.append(nn.Linear(layer_size, layer_size))
            self.model.append(nn.ReLU())
            self.model.append(nn.Dropout(0.2))

        # Setup output layer
        self.model.append(nn.Linear(layer_size, 1))

    def forward(self, x):
        return torch.squeeze(self.model(x))

    def train(self, train_data, loss_fn, learning_rate):
        # Train funktion for simple shapes, currently probably deprecated
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
        # Test function for simple shapes, currently probably deprecated
        self.model.eval()
        num_batches = len(test_data)
        test_loss = 0

        with torch.no_grad():
            for X, y in test_data:
                pred = torch.squeeze(self.model(X))
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f"Test Error: \n Avg loss: {test_loss:>8f}\n")

    def train_multi_shape(self,
                          shapes: list,
                          coord_sdf_tuple: list,
                          samples: int,
                          epochs: int,
                          learning_rate_parameters: float,
                          learning_rate_latent: float) -> list:
        """
            @brief: trains the model on multiple Meshes
            @inputs:
                shapes: a list of trimeshes
                coord_sdf_tuple: a list of coord sdf pairs for each shape
                samples: how many points to sample from each mesh
                epochs: over how many epochs the network is trained
                learning_rate_parameters: the rate with which the model parameters are updated
                learning_rate_latent: the rate with which the latent vectors get updated

            @output:
                List of latent Vectors in the same order as the input meshes
        """
        self.model.train()
        # sample provided models
        number_of_shapes = len(shapes)

        # create inital latent vectors
        mul_norm_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(self.latent_size), torch.eye(self.latent_size)* SIGMA**2)
        latent_vectors = [mul_norm_dist.sample() for _ in range(number_of_shapes)]

        # define optimizers for model and latent space
        optimizer_latent = torch.optim.Adam(latent_vectors,lr=learning_rate_latent)

        optimizer_parameter = torch.optim.Adam(self.model.parameters(),
                                               lr=learning_rate_parameters)

        for i in range(epochs):
            loss = 0
            # TODO: Implement batch procedre in order to train more than a handfull of shapes
            for latent, (coord, sdf) in zip(latent_vectors, coord_sdf_tuple):
                coord = torch.from_numpy(coord)
                sdf = torch.from_numpy(sdf)
                
                # Create input Tensor
                input_tensor = torch.column_stack((latent.expand(samples,self.latent_size), coord))


                pred = torch.squeeze(self.model(input_tensor))
                regulisation_factor = (1/SIGMA**2) * torch.linalg.norm(latent)
                loss += self.loss_fn(pred, sdf) + regulisation_factor
            loss.backward()
            optimizer_latent.step()
            optimizer_latent.zero_grad()
            optimizer_parameter.step()
            optimizer_parameter.zero_grad()
            if i % 100 == 0:
                print(f"loss: {loss:>7f}, epoch: {i:>5d}")

        self.model.eval()
        return [self.inferece_latentent(coord, sdf) for coord, sdf in coord_sdf_tuple]

    def inferece_latentent(self, coord, sdf, epochs: int=5, lr: float=0.001):
        # first guess for latent
        latent = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(self.latent_size), torch.eye(self.latent_size)*SIGMA**2)
        latent = [torch.squeeze(latent.sample_n(1))]
        
        # initilize optimizer
        optimizer = torch.optim.Adam(latent)

        # transform inputs into tensors
        coord = torch.from_numpy(coord)
        sdf = torch.from_numpy(sdf)

        self.model.eval()
        # optimize latent space
        for _ in range(epochs):
            input_tensor = torch.column_stack((latent[0].expand(coord.shape[0], self.latent_size),coord))
            pred = torch.squeeze(self.model(input_tensor))
            reg_term = (1/SIGMA**2) * torch.linalg.norm(latent[0])
            loss = self.loss_fn(pred, sdf) + reg_term

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return latent


def ssd_Loss(x, y):
    delta = torch.full_like(x, TOL)
    clamp_x = torch.minimum(delta, torch.maximum(-delta, x))
    clamp_y = torch.minimum(delta, torch.maximum(-delta, y))

    l1_loss = nn.L1Loss(reduction='sum')
    return l1_loss(clamp_x, clamp_y)


def chanfer_distance(coord_list_1: npt.ArrayLike,
                     coord_list_2: npt.ArrayLike) -> float:
    return 0.0


if __name__ == "__main__":

    inst = msdSDF(64, 5, 16, ssd_Loss)

    path_list = [
        "ModelNet10\\ModelNet10\\desk\\train\\desk_0053.off",
        "ModelNet10\\ModelNet10\\monitor\\train\\monitor_0002.off",
        "ModelNet10\\ModelNet10\\bathtub\\train\\bathtub_0016.off"
    ]

    mesh_list = [trimesh.load_mesh(path) for path in path_list]

    latent = inst.train_multi_shape(
        shapes=mesh_list,
        samples=16384,
        epochs=200,
        learning_rate_latent=0.001,
        learning_rate_parameters=0.00003
    )

    print(latent)
    

