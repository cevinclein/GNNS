import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pygame

SDF_MODEL = None
MAX_DIST = 100.0
MAX_STEPS = 200
EPSILON = 0.0001

class ConditionalSDFVAE(nn.Module):
    def __init__(self, num_functions=2, latent_size=16, input_dim=3, hidden_size=64, num_hidden_layers=3):
        """
        Conditional SDF VAE Model.
        
        Args:
            num_functions (int): Number of SDF functions (e.g. 2: sphere and box).
            latent_size (int): Dimension of the latent vector.
            input_dim (int): Dimension of the input coordinate (typically 3).
            hidden_size (int): Number of hidden units in each hidden layer.
            num_hidden_layers (int): Number of hidden layers in each decoder branch.
        """
        super(ConditionalSDFVAE, self).__init__()
        self.num_functions = num_functions
        self.latent_size = latent_size
        self.input_dim = input_dim

        # Encoder: Instead of a single embedding, we now have two embeddings that map a function label
        # to the parameters of a latent distribution: mean and log-variance.
        self.encoder_mu = nn.Embedding(num_functions, latent_size)
        self.encoder_logvar = nn.Embedding(num_functions, latent_size)

        # Decoder: Create one branch (an MLP) per function.
        # Each branch takes as input the concatenated coordinate (3D) and latent vector, and outputs a single float.
        self.decoders = nn.ModuleList([
            self._build_decoder(input_dim + latent_size, hidden_size, num_hidden_layers)
            for _ in range(num_functions)
        ])
    
    def _build_decoder(self, in_features, hidden_size, num_hidden_layers):
        layers = []
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_features if i == 0 else hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))  # Final layer outputs a single float.
        return nn.Sequential(*layers)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample_latent(self, batch_size, device='cpu'):
        """
        Sample a latent vector from the prior (standard normal).
        
        Args:
            batch_size (int): Number of latent vectors to sample.
            device (str): Device on which to create the latent tensor.
        
        Returns:
            torch.Tensor: A tensor of shape (batch_size, latent_size) sampled from N(0,I).
        """
        return torch.randn(batch_size, self.latent_size, device=device)
    
    def forward(self, x, labels, latent_override=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input coordinates, shape (batch_size, input_dim).
            labels (torch.Tensor): Function label indices, shape (batch_size,).
            latent_override (torch.Tensor, optional): If provided, uses this latent vector instead of sampling.
        
        Returns:
            tuple:
                - outputs (torch.Tensor): Predicted SDF values, shape (batch_size, 1).
                - kl_loss (torch.Tensor): KL divergence loss (scalar).
        """
        batch_size = x.size(0)
        if latent_override is None:
            # Get latent distribution parameters from the encoder based solely on the label.
            mu = self.encoder_mu(labels)        # Shape: (batch_size, latent_size)
            logvar = self.encoder_logvar(labels)  # Shape: (batch_size, latent_size)
            # Sample latent code using the reparameterization trick.
            z = self.reparameterize(mu, logvar)
            # Compute KL divergence loss per sample and average over the batch.
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        else:
            z = latent_override
            kl_loss = torch.tensor(0.0, device=x.device)

        # Concatenate the latent vector with the input coordinate.
        x_in = torch.cat([x, z], dim=1)  # Shape: (batch_size, input_dim + latent_size)
        outputs = []
        # Process each sample individually so we can select the correct branch.
        for i in range(batch_size):
            branch_idx = labels[i].item()
            decoder = self.decoders[branch_idx]
            xi = x_in[i].unsqueeze(0)  # Shape: (1, input_dim + latent_size)
            out = decoder(xi)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0)
        return outputs, kl_loss

    def train_model(self, train_loader, num_epochs=20, learning_rate=1e-3, kl_weight=1.0, device='cpu'):
        """
        Trains the model with a combined reconstruction and KL divergence loss.
        
        Args:
            train_loader (DataLoader): Provides tuples (x, labels, y) where:
                x: (batch_size, input_dim)
                labels: (batch_size,) with each entry a function label
                y: (batch_size, 1) true SDF values.
            num_epochs (int): Number of epochs.
            learning_rate (float): Learning rate.
            kl_weight (float): Weight for the KL divergence term.
            device (str): 'cpu' or 'cuda'.
        """
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        mse_criterion = nn.MSELoss()
        self.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                x, labels, y = batch
                x, labels, y = x.to(device), labels.to(device), y.to(device)
                optimizer.zero_grad()
                outputs, kl_loss = self.forward(x, labels)
                recon_loss = mse_criterion(outputs, y)
                loss = recon_loss + kl_weight * kl_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x.size(0)
            avg_loss = epoch_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {avg_loss:.4f}")
    
    def test_model(self, test_loader, kl_weight=1.0, device='cpu'):
        """
        Evaluates the model.
        
        Args:
            test_loader (DataLoader): Provides tuples (x, labels, y) for evaluation.
            kl_weight (float): Weight for the KL divergence term.
            device (str): 'cpu' or 'cuda'.
        """
        self.to(device)
        self.eval()
        total_loss = 0.0
        mse_criterion = nn.MSELoss()
        with torch.no_grad():
            for batch in test_loader:
                x, labels, y = batch
                x, labels, y = x.to(device), labels.to(device), y.to(device)
                outputs, kl_loss = self.forward(x, labels)
                recon_loss = mse_criterion(outputs, y)
                loss = recon_loss + kl_weight * kl_loss
                total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(test_loader.dataset)
        print(f"Test Loss: {avg_loss:.4f}")

# -------------------------------------------------------------------
# Define true SDF functions for a sphere and a box.
def sphere_sdf(point, radius=1.0):
    """
    True SDF for a sphere: distance from point to the sphere surface.
    SDF = norm(point) - radius.
    """
    return np.linalg.norm(point) - radius

def box_sdf(point, half_extents=np.array([1.0, 1.0, 1.0])):
    """
    True SDF for an axis-aligned box.
    This implementation follows:
       sdf = norm(max(|point| - half_extents, 0)) + min(max(|point| - half_extents), 0)
    """
    q = np.abs(point) - half_extents
    q_clamped = np.maximum(q, 0)
    outside_distance = np.linalg.norm(q_clamped)
    inside_distance = np.minimum(np.max(q), 0)
    return outside_distance + inside_distance

# -------------------------------------------------------------------
# Data generation: create synthetic training examples using the two true SDFs.
def generate_sdf_dataset(num_samples_per_function=500, input_range=2.0):
    points_list = []
    labels_list = []
    sdf_values_list = []
    
    for label in range(2):  # 0: sphere, 1: box
        for _ in range(num_samples_per_function):
            point = np.random.uniform(-input_range, input_range, size=(3,))
            if label == 0:
                sdf_value = sphere_sdf(point, radius=1.0)
            else:
                sdf_value = box_sdf(point, half_extents=np.array([1.0, 1.0, 1.0]))
            points_list.append(point)
            labels_list.append(label)
            sdf_values_list.append([sdf_value])  # wrap in list to have shape (1,)
    
    points = np.array(points_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)
    sdf_values = np.array(sdf_values_list, dtype=np.float32)
    return points, labels, sdf_values

def gen_Model():
    # For reproducibility.
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic SDF dataset.
    points, labels, sdf_values = generate_sdf_dataset(num_samples_per_function=500, input_range=2.0)
    
    # Create a TensorDataset and DataLoaders.
    dataset = TensorDataset(torch.from_numpy(points), torch.from_numpy(labels), torch.from_numpy(sdf_values))
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Initialize the conditional SDF VAE model.
    model = ConditionalSDFVAE(num_functions=2, latent_size=16, input_dim=3, hidden_size=64, num_hidden_layers=3)
    
    # Train the model.
    model.train_model(train_loader, num_epochs=60, learning_rate=1e-3, kl_weight=1.0, device='cpu')
    
    # Evaluate on the test set.
    model.test_model(test_loader, kl_weight=1.0, device='cpu')
    
    # ----------------------------------------------------------------
    # Example of calling the model in two different ways:
    # 1. Using the encoder to generate the latent vector (with VAE sampling).
    test_point = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
    test_label = torch.tensor([0], dtype=torch.long)  # 0 for sphere
    
    output_using_encoder, _ = model(test_point, test_label)
    
    # 2. Sampling a latent vector from the prior and using it to override the encoder output.
    custom_latent = model.sample_latent(batch_size=1, device='cpu')
    output_using_sampled_latent, _ = model(test_point, test_label, latent_override=custom_latent)
    
    print("Output using encoder (VAE) latent:", output_using_encoder.item())
    print("Output using sampled latent:", output_using_sampled_latent.item())
    
    global SDF_MODEL
    SDF_MODEL = model

gen_Model()

# =====================================================================================================

# Here we call the SDF, label 0 is the sphere and label 1 is the box
def SDF(p, label = 0):
    x = torch.tensor([p], dtype=torch.float32)
    test_label = torch.tensor([label], dtype=torch.long)
    
    custom_latent = SDF_MODEL.sample_latent(batch_size=1, device='cpu')
    x, _ = SDF_MODEL(x, test_label, custom_latent)
    x = x.item() #detach().numpy()
    return x
    

def trace(ro, rd):
    """
    Ray-march from ray origin 'ro' along direction 'rd'.
    
    Parameters:
      ro: Ray origin as a NumPy array (3,)
      rd: Normalized ray direction (3,)
      
    Returns:
      A tuple (depth, steps) where:
         depth: The distance along the ray where the hit was detected.
                If no hit is found within MAX_STEPS, returns MAX_DIST.
         steps: A measure based on the number of remaining steps (used for shading).
    """
    depth = 0.0
    for i in range(MAX_STEPS):
        p = ro + depth * rd
        d = SDF(p)
        if d < EPSILON:
            steps = MAX_STEPS - i  # as in shader: steps = 200 - i
            return depth, steps
        depth += d
        if depth > MAX_DIST:
            return MAX_DIST, MAX_STEPS - i
    return MAX_DIST, 0

def render_scene(width, height):
    """
    Renders the scene by iterating over every pixel, computing a ray direction,
    ray marching into the scene, and returning an image array of shape (height, width, 3).
    
    The ray setup follows:
      - fragCoord: pixel coordinate.
      - xy = (fragCoord - (iResolution/2)) / iResolution.y,
      - rd = normalize(vec3(xy, -1)).
      - ro is set to (0, 0, 4.5) (camera position).
      
    The pixel color is computed in grayscale: if a hit is detected the intensity is proportional
    to (steps / MAX_STEPS), otherwise black.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # Camera setup
    ro = np.array([0.0, 0.0, 4.5])
    
    for y in range(height):
        for x in range(width):
            # Convert pixel coordinate to normalized coordinate,
            # center at (width/2, height/2) and divide by height.
            xy = np.array([(x - width / 2) / height,
                           (y - height / 2) / height])
            # Construct the ray direction and normalize.
            rd = np.array([xy[0], xy[1], -1.0])
            rd = rd / np.linalg.norm(rd)
            
            depth, steps = trace(ro, rd)
            if depth < MAX_DIST:
                # Compute brightness based on steps.
                brightness = np.clip(steps / MAX_STEPS, 0.0, 1.0) * 255
                color = (int(brightness), int(brightness), int(brightness))
            else:
                color = (0, 0, 0)
            image[y, x] = color
        # (Optional) print progress
        if y % 20 == 0:
            print(f"Rendered {y}/{height} rows")
    return image

def main():
    # Initialize Pygame.
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("3D Ray Marching (Pygame)")
    
    # Render the scene into an image array.
    print("Rendering scene...")
    image = render_scene(width, height)
    print("Rendering complete.")
    
    # Pygame's surfarray expects an array with shape (width, height, 3),
    # so we transpose the image.
    surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
    
    # Main loop: display the rendered image until the user closes the window.
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.blit(surface, (0, 0))
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()