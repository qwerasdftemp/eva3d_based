import torch
# from pytorch3d.ops import gram_schmidt, normalize_vectors

def barycentric_coordinates_pytorch3d(points, surface):
    # Get the vertices of the triangular surface
    v0, v1, v2 = surface[:, 0, :], surface[:, 1, :], surface[:, 2, :]

    # Compute the edges of the triangle
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Compute the normal of the triangular surface using cross product
    # normal = edge1 x edge2
    normal = torch.cross(edge1, edge2, dim=1)

    # Compute the vector from the triangle's vertices to the points
    pv = points - v0

    # Project the points onto the surface by computing the dot product between
    # the normal and pv, and then subtracting the scaled normal from the points
    # projected_points = points - (dot(pv, normal) * normal)
    dot = torch.sum(pv * normal, dim=1)
    projected_points = points - dot.unsqueeze(-1) * normal

    # Compute the barycentric coordinates using Cramer's rule
    # First, compute the dot products needed for the system of linear equations
    d00 = torch.sum(edge1 * edge1, dim=1)
    d01 = torch.sum(edge1 * edge2, dim=1)
    d11 = torch.sum(edge2 * edge2, dim=1)
    d20 = torch.sum((projected_points - v0) * edge1, dim=1)
    d21 = torch.sum((projected_points - v0) * edge2, dim=1)

    # Compute the determinants using Cramer's rule
    denom = d00 * d11 - d01 * d01
    bary_v = (d00 * d21 - d01 * d20) / denom
    bary_w = (d11 * d20 - d01 * d21) / denom
    bary_u = 1.0 - bary_v - bary_w

    # Combine the barycentric coordinates into a single tensor
    barycentric_coords = torch.stack([bary_u, bary_v, bary_w], dim=1)

    return barycentric_coords

# Example usage:
points = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

surface = torch.tensor([
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ],
    [
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0]
    ]
])

barycentric_coords = barycentric_coordinates_pytorch3d(points, surface)
print(barycentric_coords)