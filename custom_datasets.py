import numpy as np
import numpy.typing as npt
import torch

BOUNDING_BOX_LENGTH = 1.0
class sdSphere(torch.utils.data.Dataset):
    def __init__(self, radius=-1.5, samples = 64000):
        self.samples = samples
        self.coords = np.asarray(np.random.default_rng().uniform(-BOUNDING_BOX_LENGTH,BOUNDING_BOX_LENGTH,(self.samples,3),), dtype=np.float32)
        self.sdf = np.linalg.norm(self.coords, axis=1) - radius
    def __len__(self):
        return self.samples
    def __getitem__(self, index):
        return self.coords[index], self.sdf[index]

class sdBox(torch.utils.data.Dataset):
    def __init__(self, dim:npt.ArrayLike, samples:int):
        self.samples = samples

        # generate coordinates
        self.coords = np.random.default_rng().uniform(-BOUNDING_BOX_LENGTH, BOUNDING_BOX_LENGTH, self.samples) 

        # convert coords into datatype usable by torch
        self.coords = np.asarray(self.coords, dtype=np.float32)

        # calculate sdf values
        q = np.abs(self.coords) - dim
        q_clip = np.maximum(q, 0.0)
        q_len = np.linalg.norm(q_clip, axis=1)
        inside = np.min(np.max(q[:,0], np.max(q[:,1], q[:,2]), 0.0))
        self.sdf = q_len + inside

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        return self.coords[index], self.sdf[index]




class sdRoundBox(torch.utils.data.Dataset):
    def __init__(self, dim:npt.ArrayLike, r:float, samples:int):
        self.samples = samples

        # generate coordinates
        self.coords = np.random.default_rng().uniform(-BOUNDING_BOX_LENGTH, BOUNDING_BOX_LENGTH, self.samples) 

        # convert coords into datatype usable by torch
        self.coords = np.asarray(self.coords, dtype=np.float32)

        # calculate sdf values
        q = np.abs(self.coords) - dim + r
        q_clip = np.maximum(q, 0.0)
        q_len = np.linalg.norm(q_clip, axis=1)
        inside = np.min(np.max(q[:,0], np.max(q[:,1], q[:,2]), 0.0))
        self.sdf = q_len + inside - r

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        return self.coords[index], self.sdf[index]

class sdMixed(torch.utils.data.Dataset):
    def __init__(self, sdf_list:list, num_samples:int):
        self.samples = num_samples * len(sdf_list)
        self.values = np.empty((0,5))

        # create latent, coord, sdf tuple for each provided sdf
        for i, sdf in enumerate(sdf_list):
            latent = np.full((num_samples,1), i)
            coords = np.random.default_rng().uniform(-BOUNDING_BOX_LENGTH, BOUNDING_BOX_LENGTH, (num_samples,3))
            coords = np.asarray(coords, dtype=np.float32)
            sdv = np.reshape(sdf(coords), (num_samples, 1))
            local_matrix = np.concatenate((latent, coords, sdv), axis=1)
            self.values = np.concatenate((self.values, local_matrix), axis=0, dtype=np.float32)

        # ensure that internal data is shufflet in order to prevent bias
        np.random.default_rng().shuffle(self.values, axis=0)

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        return self.values[index,0:-1], torch.tensor(self.values[index,-1])

class box_sdf():
    def __init__(self, box_extend:npt.ArrayLike):
        self.box_dimensions = box_extend

    def get_sdf(self, coord_array:npt.ArrayLike) -> npt.ArrayLike:
        # Calculate sdf for a box
        q = np.abs(coord_array) - self.box_dimensions
        q_clip = np.maximum(q, 0.0)
        q_len = np.linalg.norm(q_clip, axis=1)
        inside = np.minimum(np.maximum(q[:,0], np.maximum(q[:,1], q[:,2])), 0.0)
        return  q_len + inside

class sphere_sdf():
    def __init__(self, radius):
        self.radius = radius

    def get_sdf(self, coord_array:npt.ArrayLike) -> npt.ArrayLike:
        return np.linalg.norm(coord_array, axis=1) - self.radius


