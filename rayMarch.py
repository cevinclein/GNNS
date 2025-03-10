import pygame
import numpy as np

MAX_DIST = 100.0
MAX_STEPS = 200
EPSILON = 0.0001

# SDF functions
def sdRoundBox(p):
    p = p + np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 1.0, 2.0])
    r = 1.3
    q = np.abs(p) - b + r
    # For each component, clamp q to zero from below
    q_clip = np.maximum(q, 0.0)
    # Euclidean length of the clamped vector
    len_q = np.linalg.norm(q_clip)
    # For points inside the box, use the maximum of q components, clamped to zero.
    inside = min(max(q[0], max(q[1], q[2])), 0.0)
    return len_q + inside - r

def sdSphere(p):
    return np.linalg.norm(p) - 1.0

def SDF(p):
    """
    Scene SDF: minimum of the rounded box and the sphere.
    """
    return min(sdRoundBox(p), sdSphere(p))

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

