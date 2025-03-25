import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pygame
import numpy as np

SDF_MODEL = None
MAX_DIST = 100.0
MAX_STEPS = 200
EPSILON = 0.0001

# Define a fully connected neural network that can be resized using parameters.
class FullyConnectedNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_hidden_layers=3, output_size=1):
        super(FullyConnectedNet, self).__init__()
        layers = []
        # First layer (input to first hidden layer)
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        # Add additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        # Output layer (hidden to output)
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Define an example signed distance function.
def signed_distance_function(point):
    x, y, z = point[0], point[1], point[2]
    return np.sqrt(x**2 + y**2 + z**2) - 1

# Generate synthetic training or testing data.
def generate_data(num_samples=1000):
    # Randomly sample points in the range [-2, 2] for each coordinate.
    X = np.random.uniform(-2, 2, (num_samples, 3))
    # Calculate the target signed distance for each point.
    y = np.array([signed_distance_function(pt) for pt in X]).reshape(-1, 1)
    # Convert to torch tensors.
    return torch.FloatTensor(X), torch.FloatTensor(y)

# Training loop for the neural network.
def train(model, X, y, epochs=1000, lr=0.001):
    criterion = nn.MSELoss()  # Mean squared error loss for regression.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# Test function to evaluate the model on test data.
def test_model(model, X_test, y_test, tolerance=0.1):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(X_test)
        mse_loss = nn.MSELoss()(predictions, y_test)
        # Calculate a simple "accuracy" metric as the fraction of predictions within a given tolerance.
        within_tolerance = torch.abs(predictions - y_test) < tolerance
        accuracy = torch.mean(within_tolerance.float()).item() * 100  # percentage
        
    print(f"\nTest Results:")
    print(f"Mean Squared Error: {mse_loss.item():.6f}")
    print(f"Accuracy (|error| < {tolerance}): {accuracy:.2f}%")
    model.train()  # Set the model back to training mode

def gen_Model():
    # Network parameters (modify these to change network size)
    input_size = 3
    hidden_size = 64       # Change to increase/decrease neurons per hidden layer.
    num_hidden_layers = 3  # Change to increase/decrease the number of hidden layers.
    output_size = 1

    # Initialize the neural network.
    model = FullyConnectedNet(input_size, hidden_size, num_hidden_layers, output_size)
    
    # Generate training data.
    X_train, y_train = generate_data(num_samples=10000)
    
    # Train the model.
    train(model, X_train, y_train, epochs=1000, lr=0.001)
    
    # Generate test data.
    X_test, y_test = generate_data(num_samples=2000)
    
    # Evaluate the model on the test data.
    test_model(model, X_test, y_test, tolerance=0.1)
    
    # Test the trained model on a sample point.
    test_point = torch.FloatTensor([[0.5, 0.5, 0.5]])
    predicted = model(test_point)
    true_value = signed_distance_function([0.5, 0.5, 0.5])
    
    print("\nTest Sample:")
    print(f"Input Point: {test_point.numpy()}")
    print(f"Predicted Signed Distance: {predicted.item():.6f}")
    print(f"True Signed Distance: {true_value:.6f}")
    
    global SDF_MODEL
    SDF_MODEL = model

gen_Model()


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