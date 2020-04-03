import numpy as np

import imgpr.consts as consts
from .node import node

class sew(node):
    def __init__(self, x, seam, sew_func, shape, direction=consts.HORIZONTAL):
        super().__init__(shape)
        assert(direction in [consts.HORIZONTAL, consts.VERTICAL])
        self._x = x
        self._direction = direction
        self._seam = seam
        if type(sew_func) is int:
            size = sew_func
            sew_func = lambda x : 1 if x > size else 0
        self._func = sew_func

    def _run(self, sess):
        image = sess.feed_dict.get(self)
        if image is not None:
            return image

        if hasattr(self._seam, 'run'):
            seam = self._seam.run(sess)
        else:
            seam = self._seam

        if hasattr(self._x, 'run'):
            return self._run_sew(self._x.run(sess), seam, self._func)
        return self._run_sew(self._x, seam, self._func)

    def _run_sew(self, image, seam, sew_func):
        ret = np.zeros(shape=(self.shape[0], self.shape[1], 3), dtype=int)
        if self._direction == consts.VERTICAL:
            image = image.transpose((1, 0, 2))
            seam = seam.T
            ret = ret.transpose((1, 0, 2))
        
        for i in range(min(ret.shape[0], image.shape[0])):
            y = 0
            for j in range(image.shape[1]):
                for _ in range(int(sew_func(seam[i, j]))):
                    if y < ret.shape[1]:
                        ret[i, y] = image[i, j]
                        y += 1

        if self._direction == consts.VERTICAL:
            ret = ret.transpose((1, 0, 2))

        return ret
