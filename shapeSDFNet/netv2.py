import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pygame

SDF_MODEL = None
MAX_DIST = 100.0
MAX_STEPS = 200
EPSILON = 0.0001

# Set device and optimize for GPU if available.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True  # Optimizes CuDNN performance.

# Define a fully connected neural network that can be resized using parameters.
class FullyConnectedNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_hidden_layers=3, output_size=1):
        """
        Args:
            input_size (int): Number of input neurons (3 for x, y, z).
            hidden_size (int): Number of neurons in each hidden layer.
            num_hidden_layers (int): Total number of hidden layers.
            output_size (int): Number of output neurons (1 for the signed distance).
        """
        super(FullyConnectedNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)




# Define an example signed distance function.
def signed_distance_function(point):
    """
    Compute the signed distance from the point to the surface of a sphere centered at the origin with radius 1.
    Positive if outside the sphere, negative if inside.

    f(x, y, z) = sqrt(x^2 + y^2 + z^2) - 1
    """
    x, y, z = point[0], point[1], point[2]
    return np.sqrt(x**2 + y**2 + z**2) - 1


import numpy as np

def point_sdf(point):
    # Define an internal list of points stored as a NumPy array.
    internal_points = np.array([
        [0.0, 0.1, 0.0],
        [0.1, 0.1, 0.0],
        [0.2, 0.1, 0.0],
        [0.3, 0.1, 0.0],
        [0.4, 0.1, 0.0],
        [0.5, 0.1, 0.0],
        [0.6, 0.1, 0.0],
        [0.7, 0.1, 0.0],
        [0.8, 0.1, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.2, 0.0],
        [0.1, 0.2, 0.0],
        [0.2, 0.2, 0.0],
        [0.3, 0.2, 0.0],
        [0.4, 0.2, 0.0],
        [0.5, 0.2, 0.0],
        [0.6, 0.2, 0.0],
        [0.7, 0.2, 0.0],
        [0.8, 0.2, 0.0],
        [0.9, 0.2, 0.0],
        [0.0, 0.3, 0.0],
        [0.1, 0.3, 0.0],
        [0.2, 0.3, 0.0],
        [0.3, 0.3, 0.0],
        [0.4, 0.3, 0.0],
        [0.5, 0.3, 0.0],
        [0.6, 0.3, 0.0],
        [0.7, 0.3, 0.0],
        [0.8, 0.3, 0.0],
        [0.9, 0.3, 0.0],
        [0.0, 0.4, 0.0],
        [0.1, 0.4, 0.0],
        [0.2, 0.4, 0.0],
        [0.3, 0.4, 0.0],
        [0.4, 0.4, 0.0],
        [0.5, 0.4, 0.0],
        [0.6, 0.4, 0.0],
        [0.7, 0.4, 0.0],
        [0.8, 0.4, 0.0],
        [0.9, 0.4, 0.0],
        [0.0, 0.5, 0.0],
        [0.1, 0.5, 0.0],
        [0.2, 0.5, 0.0],
        [0.3, 0.5, 0.0],
        [0.4, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.6, 0.5, 0.0],
        [0.7, 0.5, 0.0],
        [0.8, 0.5, 0.0],
        [0.9, 0.5, 0.0],
        
        [0.0, 0.1, 0.1],
        [0.1, 0.1, 0.1],
        [0.2, 0.1, 0.1],
        [0.3, 0.1, 0.1],
        [0.4, 0.1, 0.1],
        [0.5, 0.1, 0.1],
        [0.6, 0.1, 0.1],
        [0.7, 0.1, 0.1],
        [0.8, 0.1, 0.1],
        [0.9, 0.1, 0.1],
        [0.0, 0.2, 0.1],
        [0.1, 0.2, 0.1],
        [0.2, 0.2, 0.1],
        [0.3, 0.2, 0.1],
        [0.4, 0.2, 0.1],
        [0.5, 0.2, 0.1],
        [0.6, 0.2, 0.1],
        [0.7, 0.2, 0.1],
        [0.8, 0.2, 0.1],
        [0.9, 0.2, 0.1],
        [0.0, 0.3, 0.1],
        [0.1, 0.3, 0.1],
        [0.2, 0.3, 0.1],
        [0.3, 0.3, 0.1],
        [0.4, 0.3, 0.1],
        [0.5, 0.3, 0.1],
        [0.6, 0.3, 0.1],
        [0.7, 0.3, 0.1],
        [0.8, 0.3, 0.1],
        [0.9, 0.3, 0.1],
        [0.0, 0.4, 0.1],
        [0.1, 0.4, 0.1],
        [0.2, 0.4, 0.1],
        [0.3, 0.4, 0.1],
        [0.4, 0.4, 0.1],
        [0.5, 0.4, 0.1],
        [0.6, 0.4, 0.1],
        [0.7, 0.4, 0.1],
        [0.8, 0.4, 0.1],
        [0.9, 0.4, 0.1],
        [0.0, 0.5, 0.1],
        [0.1, 0.5, 0.1],
        [0.2, 0.5, 0.1],
        [0.3, 0.5, 0.1],
        [0.4, 0.5, 0.1],
        [0.5, 0.5, 0.1],
        [0.6, 0.5, 0.1],
        [0.7, 0.5, 0.1],
        [0.8, 0.5, 0.1],
        [0.9, 0.5, 0.1],
        
        [0.0, 0.1, 0.2],
        [0.1, 0.1, 0.2],
        [0.2, 0.1, 0.2],
        [0.3, 0.1, 0.2],
        [0.4, 0.1, 0.2],
        [0.5, 0.1, 0.2],
        [0.6, 0.1, 0.2],
        [0.7, 0.1, 0.2],
        [0.8, 0.1, 0.2],
        [0.9, 0.1, 0.2],
        [0.0, 0.2, 0.2],
        [0.1, 0.2, 0.2],
        [0.2, 0.2, 0.2],
        [0.3, 0.2, 0.2],
        [0.4, 0.2, 0.2],
        [0.5, 0.2, 0.2],
        [0.6, 0.2, 0.2],
        [0.7, 0.2, 0.2],
        [0.8, 0.2, 0.2],
        [0.9, 0.2, 0.2],
        [0.0, 0.3, 0.2],
        [0.1, 0.3, 0.2],
        [0.2, 0.3, 0.2],
        [0.3, 0.3, 0.2],
        [0.4, 0.3, 0.2],
        [0.5, 0.3, 0.2],
        [0.6, 0.3, 0.2],
        [0.7, 0.3, 0.2],
        [0.8, 0.3, 0.2],
        [0.9, 0.3, 0.2],
        [0.0, 0.4, 0.2],
        [0.1, 0.4, 0.2],
        [0.2, 0.4, 0.2],
        [0.3, 0.4, 0.2],
        [0.4, 0.4, 0.2],
        [0.5, 0.4, 0.2],
        [0.6, 0.4, 0.2],
        [0.7, 0.4, 0.2],
        [0.8, 0.4, 0.2],
        [0.9, 0.4, 0.2],
        [0.0, 0.5, 0.2],
        [0.1, 0.5, 0.2],
        [0.2, 0.5, 0.2],
        [0.3, 0.5, 0.2],
        [0.4, 0.5, 0.2],
        [0.5, 0.5, 0.2],
        [0.6, 0.5, 0.2],
        [0.7, 0.5, 0.2],
        [0.8, 0.5, 0.2],
        [0.9, 0.5, 0.2],
        
        [0.0, 0.1, 0.3],
        [0.1, 0.1, 0.3],
        [0.2, 0.1, 0.3],
        [0.3, 0.1, 0.3],
        [0.4, 0.1, 0.3],
        [0.5, 0.1, 0.3],
        [0.6, 0.1, 0.3],
        [0.7, 0.1, 0.3],
        [0.8, 0.1, 0.3],
        [0.9, 0.1, 0.3],
        [0.0, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        [0.2, 0.2, 0.3],
        [0.3, 0.2, 0.3],
        [0.4, 0.2, 0.3],
        [0.5, 0.2, 0.3],
        [0.6, 0.2, 0.3],
        [0.7, 0.2, 0.3],
        [0.8, 0.2, 0.3],
        [0.9, 0.2, 0.3],
        [0.0, 0.3, 0.3],
        [0.1, 0.3, 0.3],
        [0.2, 0.3, 0.3],
        [0.3, 0.3, 0.3],
        [0.4, 0.3, 0.3],
        [0.5, 0.3, 0.3],
        [0.6, 0.3, 0.3],
        [0.7, 0.3, 0.3],
        [0.8, 0.3, 0.3],
        [0.9, 0.3, 0.3],
        [0.0, 0.4, 0.3],
        [0.1, 0.4, 0.3],
        [0.2, 0.4, 0.3],
        [0.3, 0.4, 0.3],
        [0.4, 0.4, 0.3],
        [0.5, 0.4, 0.3],
        [0.6, 0.4, 0.3],
        [0.7, 0.4, 0.3],
        [0.8, 0.4, 0.3],
        [0.9, 0.4, 0.3],
        [0.0, 0.5, 0.3],
        [0.1, 0.5, 0.3],
        [0.2, 0.5, 0.3],
        [0.3, 0.5, 0.3],
        [0.4, 0.5, 0.3],
        [0.5, 0.5, 0.3],
        [0.6, 0.5, 0.3],
        [0.7, 0.5, 0.3],
        [0.8, 0.5, 0.3],
        [0.9, 0.5, 0.3],
        
        [0.0, 0.1, 0.4],
        [0.1, 0.1, 0.4],
        [0.2, 0.1, 0.4],
        [0.3, 0.1, 0.4],
        [0.4, 0.1, 0.4],
        [0.5, 0.1, 0.4],
        [0.6, 0.1, 0.4],
        [0.7, 0.1, 0.4],
        [0.8, 0.1, 0.4],
        [0.9, 0.1, 0.4],
        [0.0, 0.2, 0.4],
        [0.1, 0.2, 0.4],
        [0.2, 0.2, 0.4],
        [0.3, 0.2, 0.4],
        [0.4, 0.2, 0.4],
        [0.5, 0.2, 0.4],
        [0.6, 0.2, 0.4],
        [0.7, 0.2, 0.4],
        [0.8, 0.2, 0.4],
        [0.9, 0.2, 0.4],
        [0.0, 0.3, 0.4],
        [0.1, 0.3, 0.4],
        [0.2, 0.3, 0.4],
        [0.3, 0.3, 0.4],
        [0.4, 0.3, 0.4],
        [0.5, 0.3, 0.4],
        [0.6, 0.3, 0.4],
        [0.7, 0.3, 0.4],
        [0.8, 0.3, 0.4],
        [0.9, 0.3, 0.4],
        [0.0, 0.4, 0.4],
        [0.1, 0.4, 0.4],
        [0.2, 0.4, 0.4],
        [0.3, 0.4, 0.4],
        [0.4, 0.4, 0.4],
        [0.5, 0.4, 0.4],
        [0.6, 0.4, 0.4],
        [0.7, 0.4, 0.4],
        [0.8, 0.4, 0.4],
        [0.9, 0.4, 0.4],
        [0.0, 0.5, 0.4],
        [0.1, 0.5, 0.4],
        [0.2, 0.5, 0.4],
        [0.3, 0.5, 0.4],
        [0.4, 0.5, 0.4],
        [0.5, 0.5, 0.4],
        [0.6, 0.5, 0.4],
        [0.7, 0.5, 0.4],
        [0.8, 0.5, 0.4],
        [0.9, 0.5, 0.4],
        
        [0.0, 0.1, 0.5],
        [0.1, 0.1, 0.5],
        [0.2, 0.1, 0.5],
        [0.3, 0.1, 0.5],
        [0.4, 0.1, 0.5],
        [0.5, 0.1, 0.5],
        [0.6, 0.1, 0.5],
        [0.7, 0.1, 0.5],
        [0.8, 0.1, 0.5],
        [0.9, 0.1, 0.5],
        [0.0, 0.2, 0.5],
        [0.1, 0.2, 0.5],
        [0.2, 0.2, 0.5],
        [0.3, 0.2, 0.5],
        [0.4, 0.2, 0.5],
        [0.5, 0.2, 0.5],
        [0.6, 0.2, 0.5],
        [0.7, 0.2, 0.5],
        [0.8, 0.2, 0.5],
        [0.9, 0.2, 0.5],
        [0.0, 0.3, 0.5],
        [0.1, 0.3, 0.5],
        [0.2, 0.3, 0.5],
        [0.3, 0.3, 0.5],
        [0.4, 0.3, 0.5],
        [0.5, 0.3, 0.5],
        [0.6, 0.3, 0.5],
        [0.7, 0.3, 0.5],
        [0.8, 0.3, 0.5],
        [0.9, 0.3, 0.5],
        [0.0, 0.4, 0.5],
        [0.1, 0.4, 0.5],
        [0.2, 0.4, 0.5],
        [0.3, 0.4, 0.5],
        [0.4, 0.4, 0.5],
        [0.5, 0.4, 0.5],
        [0.6, 0.4, 0.5],
        [0.7, 0.4, 0.5],
        [0.8, 0.4, 0.5],
        [0.9, 0.4, 0.5],
        [0.0, 0.5, 0.5],
        [0.1, 0.5, 0.5],
        [0.2, 0.5, 0.5],
        [0.3, 0.5, 0.5],
        [0.4, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.6, 0.5, 0.5],
        [0.7, 0.5, 0.5],
        [0.8, 0.5, 0.5],
        [0.9, 0.5, 0.5]   
    ])
    
    # Ensure the input point is a NumPy array.
    point = np.asarray(point)  
    differences = internal_points - point
    distances = np.linalg.norm(differences, axis=1)
    
    return np.min(distances)





# Generate synthetic training or testing data.
def generate_data(num_samples=1000):
    # Randomly sample points in the range [-2, 2] for each coordinate.
    X = np.random.uniform(-2, 2, (num_samples, 3))
    # Calculate the target signed distance for each point.
    y = np.array([point_sdf(pt) for pt in X]).reshape(-1, 1)
    # Convert to torch tensors.
    return torch.FloatTensor(X), torch.FloatTensor(y)

# Training loop for the neural network using mini-batches.
def train(model, X, y, epochs=1000, lr=0.001, batch_size=32):
    criterion = nn.MSELoss()  # Mean squared error loss for regression.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create a DataLoader for batch processing.
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # Adjust the number of workers as needed.
        pin_memory=True if device.type=='cuda' else False
    )
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            # Transfer data to GPU (if available) with non_blocking transfers.
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= len(dataset)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")

# Test function to evaluate the model on test data.
def test_model(model, X_test, y_test, tolerance=0.1):
    model.eval()  # Set the model to evaluation mode.
    # Transfer test data to device.
    X_test = X_test.to(device, non_blocking=True)
    y_test = y_test.to(device, non_blocking=True)
    
    with torch.no_grad():
        predictions = model(X_test)
        mse_loss = nn.MSELoss()(predictions, y_test)
        # Calculate a simple "accuracy" metric as the fraction of predictions within a given tolerance.
        within_tolerance = torch.abs(predictions - y_test) < tolerance
        accuracy = torch.mean(within_tolerance.float()).item() * 100  # percentage.
    
    print(f"\nTest Results:")
    print(f"Mean Squared Error: {mse_loss.item():.6f}")
    print(f"Accuracy (|error| < {tolerance}): {accuracy:.2f}%")
    model.train()  # Set back to training mode.

def gen_model():
    # Network parameters (modify these to change network size).
    input_size = 3
    hidden_size = 64       # Change to increase/decrease neurons per hidden layer.
    num_hidden_layers = 3  # Change to increase/decrease the number of hidden layers.
    output_size = 1

    # Training parameters.
    batch_size = 32        # Adjust the batch size as desired.
    epochs = 1000
    learning_rate = 0.001

    # Initialize the neural network and move it to the device.
    model = FullyConnectedNet(input_size, hidden_size, num_hidden_layers, output_size).to(device)
    
    # Generate training and testing data.
    X_train, y_train = generate_data(num_samples=10000)
    X_test, y_test = generate_data(num_samples=2000)
    
    # Train the model.
    train(model, X_train, y_train, epochs=epochs, lr=learning_rate, batch_size=batch_size)
    
    # Evaluate the model on the test data.
    test_model(model, X_test, y_test, tolerance=0.1)
    
    # Test the trained model on a sample point.
    test_point = torch.FloatTensor([[0.5, 0.5, 0.5]]).to(device)
    with torch.no_grad():
        predicted = model(test_point)
    true_value = point_sdf([0.5, 0.5, 0.5])
    
    print("\nTest Sample:")
    print(f"Input Point: {test_point.cpu().numpy()}")
    print(f"Predicted Signed Distance: {predicted.item():.6f}")
    print(f"True Signed Distance: {true_value:.6f}")
    
    global SDF_MODEL
    SDF_MODEL = model

gen_model()


def SDF(p):
    x = torch.tensor(p, dtype=torch.float)
    x = SDF_MODEL(x)
    x = x.detach().numpy()
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