import numpy as np

def point_trans(image, points):
    pts = np.array(points, dtype=float)
    pts[:,0] *= 1 / (image.shape[0] - 1)
    pts[:,1] *= 1 / (image.shape[1] - 1)

    pts[:, [0, 1]] = pts[:, [1, 0]]
    return pts
