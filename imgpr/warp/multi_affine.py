import numpy as np
from scipy.spatial import Delaunay

from .utils import fix
from .affine import affine_matrix

class MultiAffine:
    def __init__(self, origin_points, new_points, triangulation=None):
        for p in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]:
            assert(p in origin_points)
            assert(p in new_points)
        if triangulation is None:
            triangulation = Delaunay(new_points)
        
        self.origin_points = np.array(origin_points)
        self.new_points = np.array(new_points)
        self.triangulation = triangulation

        self.affines = []

        for triangle in triangulation.simplices:
            mat1 = affine_matrix(*self.origin_points[triangle])
            mat2 = affine_matrix(*self.new_points[triangle])
            affine = np.matmul(np.linalg.inv(mat2), mat1)
            self.affines.append(affine)

    def __call__(self, x, y, origin_width, origin_height, new_width, new_height):
        nx = x / new_width
        ny = y / new_height

        index = self.triangulation.find_simplex((nx, ny))
        assert(index != -1)
        nx, ny, z = np.matmul(np.array([nx, ny, 1]), self.affines[index])

        # print(x, y, nx / z * origin_width, ny / z * origin_height)

        return fix(nx / z * origin_width, ny / z * origin_height, origin_width, origin_height)
