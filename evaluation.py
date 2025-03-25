import multi_shape_deepSDF
from scipy.optimize import linear_sum_assignment
import numpy as np
import scipy
import torch
import os
import trimesh
import plotly.graph_objects as go
from mesh_to_sdf import sample_sdf_near_surface
from sdf.sdf import *
import skimage

PATH = "temp\\model_size_eval.pt"
DEFAULT_HIDDEN_LAYER_NUMBER = 4
DEFAULT_HIDDEN_LAYER_SIZE = 32
DEFAULT_LATENT_SIZE = 8
LR_LATENT = 0.001
LR_PARAMETERS = 0.0003
SAMPLES = 16384
LOSS_FN = multi_shape_deepSDF.ssd_Loss
EPOCHS = 1000
MESH_PATHS = [
    "ModelNet10\\ModelNet10\\monitor\\train\\monitor_0001.off",
    "ModelNet10\\ModelNet10\\bathtub\\train\\bathtub_0001.off",
    "ModelNet10\\ModelNet10\\sofa\\train\\sofa_0001.off"
]
SHAPES = [trimesh.load(path) for path in MESH_PATHS]
USHAPES = [mesh.apply_scale(1.0/mesh.extents) for mesh in SHAPES]
COORD_SDF_PAIRS = [sample_sdf_near_surface(shape, number_of_points=SAMPLES) for shape in USHAPES]
SIZE_MESH = sum([os.path.getsize(path) for path in MESH_PATHS])
EMD_SAMPLES = 250


def get_mesh(model, latent_vec, steps):
    spacing = np.linspace(-1.0,1.0,steps, dtype=np.float32)
    xx, yy, zz = np.meshgrid(spacing, spacing, spacing)
    voxel_list = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())))
    input_vec = torch.column_stack((latent_vec.expand(voxel_list.shape[0], latent_vec.shape[0]), voxel_list))
    voxel_sdf = model(input_vec).detach().numpy().reshape((steps, steps, steps))
    vertices, faces, normals, _ = skimage.measure.marching_cubes(voxel_sdf)
    pred_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return pred_mesh

def save_pred_model(model, latent_vec, str):
    folder_path = f'results\\{str}'
    try:
        os.makedirs(folder_path)
    except:
        pass
    for latent, mesh_name in zip(latent_vec, MESH_PATHS):
        pred_mesh = get_mesh(model, latent[0],50)
        png = pred_mesh.scene().save_image()
        with open(folder_path + f'\\{os.path.basename(mesh_name)}.png', 'wb') as img:
            img.write(png)
        pred_mesh.export(folder_path + f'\\{os.path.basename(mesh_name)}.glb')


def emd_calculation(pc_1, pc_2) -> float:
        """
        @brief Compute the Earth Mover's Distance (EMD) between the two point clouds 
               (assuming equal number of points and uniform mass)
        @return The EMD distance (float)
        
        @note EMD computed using the Hungarian algorithm (linear sum assignment)
              This assumes both point clouds have the same number of points, and
              each point has equal "weight". If sizes differ, we match up to the min of both
        """
        nA = len(pc_1)
        nB = len(pc_2)
        n  = min(nA, nB)
        
        pcA = pc_1[:n]
        pcB = pc_2[:n]
        
        cost_matrix = np.zeros((n, n))

        for i in range(n):
            cost_matrix[i] = np.linalg.norm(pcB - pcA[i], axis=1)
        
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        total_cost       = cost_matrix[row_idx, col_idx].sum()
        
        # RETURN AVERAGE COST
        return total_cost / n


def earth_mover_distance(model, latent_vec) -> float:

    emd = np.empty(len(latent_vec))
    for d, latent, true_mesh in zip(emd, latent_vec, SHAPES):
        try:
            pred_mesh = get_mesh(model, latent[0], 50)
        
            point_cloud_1 = pred_mesh.sample(EMD_SAMPLES)
            point_cloud_2 = true_mesh.sample(EMD_SAMPLES)
            d = scipy.stats.wasserstein_distance_nd(point_cloud_1, point_cloud_2)
            d = emd_calculation(point_cloud_1, point_cloud_2)
        except TypeError:
            print("Scipy failed")
        except:
            print("MC Error, probably no surface found")
            d = np.nan

    return emd


def champfer_distance(model, latent_vec) -> float:
    champfer_dist = np.empty(len(latent_vec))
    for cd, latent, true_mesh in zip(champfer_dist, latent_vec, SHAPES):

        try:
            pred_mesh = get_mesh(model, latent[0], 50)
        
            point_cloud_1 = pred_mesh.sample(EMD_SAMPLES)
            point_cloud_2 = true_mesh.sample(EMD_SAMPLES)
            distance = 0
            for point in point_cloud_1:
                distance += np.min(np.linalg.norm(point - point_cloud_2))

            for point in point_cloud_2:
                distance += np.min(np.linalg.norm(point - point_cloud_1))
            cd = distance
        except:
            print("MC Error, probably no surface found")


    return champfer_dist


def model_pickel_size(model) -> int:
    torch.save(model, PATH)
    size = os.path.getsize(PATH)
    os.remove(PATH)
    return size


def model_state_dict_size(model) -> int:
    torch.save(model.state_dict(), PATH)
    size = os.path.getsize(PATH)
    os.remove(PATH)
    return size


def eval_hidden_layer_number(values):
    pickel_size = []
    stat_dict_size = []
    emd = []
    champfer = []
    # eval model for each possible value
    for val in values:
        print(f'Evaluating model with {val} hidden layers\n')
        eval_model = multi_shape_deepSDF.msdSDF(layer_number=val,
                                                latent_size=DEFAULT_LATENT_SIZE,
                                                layer_size=DEFAULT_HIDDEN_LAYER_SIZE,
                                                loss_fn=LOSS_FN)
        latent = eval_model.train_multi_shape(shapes=SHAPES,
                                              coord_sdf_tuple=COORD_SDF_PAIRS,
                                              samples=SAMPLES,
                                              epochs=EPOCHS,
                                              learning_rate_latent=LR_LATENT,
                                              learning_rate_parameters=LR_PARAMETERS)

        emd.append(earth_mover_distance(eval_model, latent))
        champfer.append(champfer_distance(eval_model, latent))
        pickel_size.append(model_pickel_size(eval_model))
        stat_dict_size.append(model_state_dict_size(eval_model))
        save_pred_model(eval_model, latent, f'hidden_number_{val}')


    # plot champfer results
    champfer = np.array(champfer)
    champfer_fig = go.Figure()

    for i, mesh_name in enumerate(MESH_PATHS):
        champfer_fig.add_trace(go.Scatter(
            x=values,
            y=champfer[:, i],
            mode='lines+markers',
            name=os.path.basename(mesh_name)
        ))
    champfer_fig.add_trace(go.Scatter(
        x=values,
        y=np.mean(champfer, axis=1),
        mode='lines+markers',
        name='Average'
    ))

    champfer_fig.update_xaxes(title_text='Number of hidden layers')
    champfer_fig.update_yaxes(title_text='Champfer distance')
    champfer_fig.update_layout(title=dict(text="Effect of hidden layer number on Champfer distance"))
    champfer_fig.show()
    # plot emd results
    emd = np.array(emd)
    emd_fig = go.Figure()

    for i, mesh_name in enumerate(MESH_PATHS):
        emd_fig.add_trace(go.Scatter(
            x=values,
            y=emd[:, i],
            mode='lines+markers',
            name=os.path.basename(mesh_name)
        ))
    emd_fig.add_trace(go.Scatter(
        x=values,
        y=np.mean(emd, axis=1),
        mode='lines+markers',
        name='Average'
    ))

    emd_fig.update_xaxes(title_text='Number of hidden layers')
    emd_fig.update_yaxes(title_text='Wasserstein distance')
    emd_fig.update_layout(title=dict(text="Effect of hidden layer number on Wasserstein distance"))
    emd_fig.show()

    # plot size results
    size_fig = go.Figure(
    )
    size_fig.add_trace(go.Scatter(
        x=values,
        y=pickel_size,
        mode='lines+markers',
        name='Pickeled'
    ))
    size_fig.add_trace(go.Scatter(
        x=values,
        y=stat_dict_size,
        mode='lines+markers',
        name='state dict'
    ))
    size_fig.add_hline(y=SIZE_MESH)
    size_fig.update_xaxes(title_text='Number of hidden layers')
    size_fig.update_yaxes(title_text='Memory size [bytes]')

    size_fig.update_layout(title=dict(text="Effect of hidden layer number on memory size"))
    size_fig.show()


# TODO: Evaluate: Hidden layer size
def eval_hidden_layer_size(values):
    pickel_size = []
    stat_dict_size = []
    emd = []
    champfer = []
    # eval model for each possible value
    for val in values:
        print(f'Evaluating model with hidden layers of size {val} \n')
        eval_model = multi_shape_deepSDF.msdSDF(layer_number=DEFAULT_HIDDEN_LAYER_NUMBER,
                                                latent_size=DEFAULT_LATENT_SIZE,
                                                layer_size=val,
                                                loss_fn=LOSS_FN)
        latent = eval_model.train_multi_shape(shapes=SHAPES,
                                     coord_sdf_tuple=COORD_SDF_PAIRS,
                                     samples=SAMPLES,
                                     epochs=EPOCHS,
                                     learning_rate_latent=LR_LATENT,
                                     learning_rate_parameters=LR_PARAMETERS)
        emd.append(earth_mover_distance(eval_model, latent))
        champfer.append(champfer_distance(eval_model, latent))
        pickel_size.append(model_pickel_size(eval_model))
        stat_dict_size.append(model_state_dict_size(eval_model))
        save_pred_model(eval_model, latent, f'hidden_size_{val}')

    # plot champfer results
    champfer = np.array(champfer)
    champfer_fig = go.Figure()

    for i, mesh_name in enumerate(MESH_PATHS):
        champfer_fig.add_trace(go.Scatter(
            x=values,
            y=champfer[:, i],
            mode='lines+markers',
            name=os.path.basename(mesh_name)
        ))
    champfer_fig.add_trace(go.Scatter(
        x=values,
        y=np.mean(champfer, axis=1),
        mode='lines+markers',
        name='Average'
    ))

    champfer_fig.update_xaxes(title_text='Size of hidden layers')
    champfer_fig.update_yaxes(title_text='Champfer distance')
    champfer_fig.update_layout(title=dict(text="Effect of hidden layer size on Champfer distance"))
    champfer_fig.show()
    # plot emd results
    emd = np.array(emd)
    emd_fig = go.Figure()

    for i, mesh_name in enumerate(MESH_PATHS):
        emd_fig.add_trace(go.Scatter(
            x=values,
            y=emd[:, i],
            mode='lines+markers',
            name=os.path.basename(mesh_name)
        ))
    emd_fig.add_trace(go.Scatter(
        x=values,
        y=np.mean(emd, axis=1),
        mode='lines+markers',
        name='Average'
    ))

    emd_fig.update_xaxes(title_text='Size of hidden layers')
    emd_fig.update_yaxes(title_text='Wasserstein distance')
    emd_fig.update_layout(title=dict(text="Effect of hidden layer size on Wasserstein distance"))
    emd_fig.show()

    # plot size results
    size_fig = go.Figure()
    size_fig.add_trace(go.Scatter(
        x=values,
        y=pickel_size,
        mode='lines+markers',
        name='Pickeled'
    ))
    size_fig.add_trace(go.Scatter(
        x=values,
        y=stat_dict_size,
        mode='lines+markers',
        name='state dict'
    ))
    size_fig.add_hline(y=SIZE_MESH)
    size_fig.update_xaxes(title_text='Size of hidden layers')
    size_fig.update_yaxes(title_text='Memory size [bytes]')
    size_fig.update_layout(title=dict(text="Effect of hidden layer size on memory size"))

    size_fig.show()


# TODO: Evaluate: Latent vector size
def eval_latent_vector_size(values):
    pickel_size = []
    stat_dict_size = []
    emd, champfer = [], []
    # eval model for each possible value
    for val in values:
        print(f'Evaluating model with latent vector of size {val} \n')
        eval_model = multi_shape_deepSDF.msdSDF(layer_number=DEFAULT_HIDDEN_LAYER_NUMBER,
                                                latent_size=val,
                                                layer_size=DEFAULT_HIDDEN_LAYER_SIZE,
                                                loss_fn=LOSS_FN)
        latent = eval_model.train_multi_shape(shapes=SHAPES,
                                     coord_sdf_tuple=COORD_SDF_PAIRS,
                                     samples=SAMPLES,
                                     epochs=EPOCHS,
                                     learning_rate_latent=LR_LATENT,
                                     learning_rate_parameters=LR_PARAMETERS)
        emd.append(earth_mover_distance(eval_model, latent))
        champfer.append(champfer_distance(eval_model, latent))
        pickel_size.append(model_pickel_size(eval_model))
        stat_dict_size.append(model_state_dict_size(eval_model))
        save_pred_model(eval_model, latent, f'latent_size_{val}')

    # plot champfer results
    champfer = np.array(champfer)
    champfer_fig = go.Figure()

    for i, mesh_name in enumerate(MESH_PATHS):
        champfer_fig.add_trace(go.Scatter(
            x=values,
            y=champfer[:, i],
            mode='lines+markers',
            name=os.path.basename(mesh_name)
        ))
    champfer_fig.add_trace(go.Scatter(
        x=values,
        y=np.mean(champfer, axis=1),
        mode='lines+markers',
        name='Average'
    ))

    champfer_fig.update_xaxes(title_text='Size of latent space')
    champfer_fig.update_yaxes(title_text='Champfer distance')
    champfer_fig.update_layout(title=dict(text="Effect of latent space size on Champfer distance"))
    champfer_fig.show()
    # plot emd results
    emd = np.array(emd)
    emd_fig = go.Figure()

    for i, mesh_name in enumerate(MESH_PATHS):
        emd_fig.add_trace(go.Scatter(
            x=values,
            y=emd[:, i],
            mode='lines+markers',
            name=os.path.basename(mesh_name)
        ))
    emd_fig.add_trace(go.Scatter(
        x=values,
        y=np.mean(emd, axis=1),
        mode='lines+markers',
        name='Average'
    ))

    emd_fig.update_xaxes(title_text='Size of latent space')
    emd_fig.update_yaxes(title_text='Wasserstein distance')
    emd_fig.update_layout(title=dict(text="Effect of latent space size on Wasserstein distance"))
    emd_fig.show()

    # plot size results
    size_fig = go.Figure()
    size_fig.add_trace(go.Scatter(
        x=values,
        y=pickel_size,
        mode='lines+markers',
        name='Pickeled'
    ))
    size_fig.add_trace(go.Scatter(
        x=values,
        y=stat_dict_size,
        mode='lines+markers',
        name='state dict'
    ))
    size_fig.add_hline(y=SIZE_MESH)
    size_fig.update_xaxes(title_text='Size of latent vector')
    size_fig.update_yaxes(title_text='Memory size [bytes]')
    size_fig.update_layout(title=dict(text="Effect of latent vector size on memory size"))

    size_fig.show()


# TODO: Evaluate: Number of Models saved

if __name__ == '__main__':
    eval_hidden_layer_number([2, 8, 16, 32])
    eval_hidden_layer_size([32, 64, 128, 256])
    eval_latent_vector_size([8, 16, 32, 64, 128])
