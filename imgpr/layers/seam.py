import numpy as np

from scipy import signal

import imgpr.consts as consts
from .node import node

sobel_x = np.c_[
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
]

sobel_y = np.c_[
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
]

class seam(node):
    def __init__(self, x, init_energy=None, iters=-1, step=1, direction=consts.HORIZONTAL):
        super().__init__(x.shape)
        assert(direction in [consts.HORIZONTAL, consts.VERTICAL])
        self._x = x
        self._direction = direction
        if iters == -1:
            if direction == consts.HORIZONTAL:
                iters = x.shape[1]
            else:
                iters = x.shape[0]
        self._iters = iters
        self._init_energy = init_energy
        self._step = step

    def _run(self, sess):
        image = sess.feed_dict.get(self)
        if image is not None:
            return image
        
        if self._init_energy is None:
            init_energy = np.zeros(shape=self.shape, dtype=int)
        elif hasattr(self._init_energy, 'run'):
            init_energy = self._init_energy.run(sess)
        else:
            init_energy = np.array(self._init_energy)
        
        assert(init_energy.shape == self.shape)

        if hasattr(self._x, 'run'):
            return self._run_seam(self._x.run(sess), init_energy)
        return self._run_seam(self._x, init_energy)

    def _run_seam(self, image, _init_energy):
        image = np.dot(image[...,:], [0.299, 0.587, 0.114])

        ret = np.ones(shape=(self.shape[0], self.shape[1]), dtype=int)
        ret *= self.shape[0] + self.shape[1]
        
        pos_map = self._init_pos()
        
        energy = np.zeros(self.shape, dtype=int)

        for t in range(self._iters):
            print(t + 1)
            energy *= 0
            now_image = np.array(pos_map, dtype=int)
            init_energy = np.array(pos_map, dtype=int)
            for x in range(len(pos_map)):
                if self._direction == consts.HORIZONTAL:
                    now_image[x, :] = image[x, now_image[x, :]]
                    init_energy[x, :] = _init_energy[x, init_energy[x, :]]
                else:
                    now_image[x, :] = image[now_image[x, :], x]
                    init_energy[x, :] = _init_energy[init_energy[x, :], x]

            sx = signal.convolve2d(now_image, sobel_x, mode="same", boundary="symm")
            sy = signal.convolve2d(now_image, sobel_y, mode="same", boundary="symm")
            energy = np.sqrt(sx * sx + sy * sy) + init_energy
            for x in range(len(pos_map) - 1):
                ori = energy[x, :]
                arr = [ori]
                lshift = ori
                rshift = ori
                for _ in range(self._step):
                    lshift = np.concatenate((lshift[1:], [np.inf]))
                    rshift = np.concatenate(([np.inf], rshift[:-1]))
                    arr.append(lshift)
                    arr.append(rshift)
                energy[x + 1, :] += np.min(arr, axis=0)
            left = 0
            right = energy.shape[1]
            for x in reversed(range(len(pos_map))):
                y = np.argmin(energy[x, left : right]) + left
                left = y - self._step
                right = y + self._step + 1
                if left < 0: left = 0
                if right > energy.shape[1]: right = energy.shape[1]
                if self._direction == consts.HORIZONTAL:
                    ret[x, pos_map[x][y]] = t + 1
                else:
                    ret[pos_map[x][y], x] = t + 1
                del pos_map[x][y]
            
        return ret

    def _init_pos(self):
        x, y = self.shape
        if self._direction == consts.VERTICAL:
            x, y = y, x
        ret = []
        for _ in range(x):
            tmp = []
            for j in range(y):
                tmp.append(j)
            ret.append(tmp)
        return ret
