import torch


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """

    # modified triplane from lpff
    return torch.tensor([
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
        [[1, 0, 0],
         [0, 0, 1],
         [0, 1, 0]],
        [[0, 0, 1],
         [0, 1, 0],
         [1, 0, 0]]], dtype=torch.float32)


def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N * n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N * n_planes, 3, 3).to("cuda")
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]


def sample_from_planes(plane_features, xyz_init, mode='nearest', padding_mode='zeros', box_warp=1):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    M, _ = xyz_init.shape
    plane_features = plane_features.view(N * n_planes, C, H, W)
    coordinates = (2 / box_warp) * xyz_init.unsqueeze(0)
    coordinates = torch.tile(coordinates, [N, 1, 1])
    plane_axes = generate_planes()
    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(
        plane_features, projected_coordinates.float(), mode=mode,
        padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features