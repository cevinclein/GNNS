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