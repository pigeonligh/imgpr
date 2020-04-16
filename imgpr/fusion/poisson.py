import numpy as np
import scipy.sparse

import imgpr.utils as utils
from .model import FusionModel

delta = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def get_sparse_matrix(mask_indicies):
    size = len(mask_indicies)
    map_indicies = {}
    for i in range(size):
        map_indicies[mask_indicies[i]] = i

    mat = scipy.sparse.lil_matrix((size, size))
    for i in range(size):
        x, y = mask_indicies[i]
        mat[i, i] = 4
        for dx, dy in delta:
            tx, ty = dx + x, dy + y
            index = map_indicies.get((tx, ty))
            if index is not None:
                mat[i, index] = -1
    return mat

def is_edge(x, y, mask):
    if mask[x, y] == 0:
        return 0
    for dx, dy in delta:
        if mask[dx + x, dy + y] == 0:
            return 1
    return 0

class PoissonFusion(FusionModel):
    def _init(self, source, mask):
        self._channels = source.shape[-1]
        mask[0, :] = 0
        mask[:, 0] = 0
        mask[-1, :] = 0
        mask[:, -1] = 0
        self._mask = mask

        non_zero = np.nonzero(mask)
        mask_indicies = list(zip(*non_zero))
        self._mask_indicies = np.array(mask_indicies)
        self._sparse_matrix = get_sparse_matrix(mask_indicies)
        self._cg_matrix = []
        for channel in range(self._channels):
            src = source[:, :, channel]

            mat = src[utils.xy2index(self._mask_indicies)] * 4
            for dx, dy in delta:
                indicies = self._mask_indicies + (dx, dy)
                mat -= src[utils.xy2index(indicies)]   
            self._cg_matrix.append(mat)
        
        self._edge = np.array([is_edge(x, y, mask) for x, y in self._mask_indicies])

    def _run_fusion(self, image):
        channels = image.shape[-1]
        result = np.zeros_like(image)
        for i in range(channels):
            result[:, :, i] = self._run_channel(i, image[:, :, i])
        return result

    def _run_channel(self, channel, image):
        assert(channel < self._channels)
        edge = image[utils.xy2index(self._mask_indicies)] * self._edge
        mat = self._cg_matrix[channel] + edge
        mask = scipy.sparse.linalg.cg(self._sparse_matrix, mat)
        result = np.zeros_like(image)
        result[utils.xy2index(self._mask_indicies)] = mask[0]
        return np.where(self._mask > 0, result, image).clip(0, 255)
