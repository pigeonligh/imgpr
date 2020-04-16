import numpy as np

"""point_trans
for MultiAffine
"""
def point_trans(image, points):
    base = [(0, 0), (image.shape[1] - 1, 0), (0, image.shape[0] - 1), (image.shape[1] - 1, image.shape[0] - 1)]
    for point in base:
        if point not in points:
            points.append(point)
    pts = np.array(points, dtype=float)
    pts[:,0] *= 1 / (image.shape[1] - 1)
    pts[:,1] *= 1 / (image.shape[0] - 1)
    return pts

def gray(image):
    return np.dot(image[...,:], [0.299, 0.587, 0.114])

def xy2index(array):
    return (array[:, 0], array[:, 1])
