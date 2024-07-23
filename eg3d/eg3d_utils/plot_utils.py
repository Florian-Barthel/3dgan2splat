import plotly.graph_objs as go
import numpy as np
import torch


def make_3d_plot(data, max_points=1000):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    np.random.shuffle(data)
    step = len(data) // max_points
    if step == 0:
        step = 1
    data = data[::step]
    return go.Figure(data=[go.Scatter3d(
        x=data[:, 0],  # X position
        y=data[:, 1],  # Y position
        z=data[:, 2],  # Z position
        mode='markers',
        marker=dict(
            size=2,  # Adjust marker size here
            opacity=0.4
        )
    )])