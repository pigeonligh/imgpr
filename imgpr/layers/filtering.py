import numpy as np

from .node import node

class filtering(node):
    def __init__(self, x, func):
        super().__init__(x.shape)

        self._x = x
        self._color_mem = {}
        self.func = func

    def _run(self, sess):
        image = sess.feed_dict.get(self)
        if image is not None:
            return image
        
        if hasattr(self._x, 'run'):
            return self._run_filtering(self._x.run(sess))
        return self._run_filtering(self._x)

    def _run_filtering(self, image):
        ret = np.zeros(shape=(self.shape[0], self.shape[1], 3), dtype=int)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                r = image[i, j, 0]
                g = image[i, j, 1]
                b = image[i, j, 2]
                color = self._color_mem.get((r, g, b))
                if color is None:
                    color = self.func(r, g, b)
                r, g, b = color
                ret[i, j, 0] = r
                ret[i, j, 1] = g
                ret[i, j, 2] = b

        return ret
