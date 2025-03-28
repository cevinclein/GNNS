import numpy as np

def sphere_sdf(point, radius=1.0):
    
    # True SDF for a sphere: distance from point to the sphere surface.
    # SDF = norm(point) - radius.
    
    return np.linalg.norm(point) - radius

def box_sdf(point, half_extents=np.array([1.0, 1.0, 1.0])):
    
    # True SDF for an axis-aligned box.
    # This implementation follows:
    # sdf = norm(max(|point| - half_extents, 0)) + min(max(|point| - half_extents), 0)
   
    q = np.abs(point) - half_extents
    q_clamped = np.maximum(q, 0)
    outside_distance = np.linalg.norm(q_clamped)
    inside_distance = np.minimum(np.max(q), 0)
    return outside_distance + inside_distance

def sd_box_frame(p):
    
    b = [0.4, 0.4, 0.4]
    e = 0.1
    
    p = np.abs(np.array(p)) - np.array(b)
    q = np.abs(p + e) - e
    
    A = (np.linalg.norm(np.maximum(np.array([p[0], q[1], q[2]]), 0.0)) +
         min(max(p[0], max(q[1], q[2])), 0.0))
    B = (np.linalg.norm(np.maximum(np.array([q[0], p[1], q[2]]), 0.0)) +
         min(max(q[0], max(p[1], q[2])), 0.0))
    C = (np.linalg.norm(np.maximum(np.array([q[0], q[1], p[2]]), 0.0)) +
         min(max(q[0], max(q[1], p[2])), 0.0))
    
    return min(min(A, B), C)