import numpy as np

from .node import node

class warping(node):
    def __init__(self, x, shape, func, fix_color=None):
        super().__init__(shape)
        shape = self.shape

        self._x = x
        self.fix_color = fix_color

        trans_matrix = np.zeros(shape=(shape[0], shape[1], 2))
        trans_flag = np.zeros(shape=shape, dtype=bool)
        for i in range(shape[0]):
            for j in range(shape[1]):
                pos_y, pos_x, flag = func(j, i, x.shape[1], x.shape[0], shape[1], shape[0])
                trans_matrix[i, j, 0] = pos_x
                trans_matrix[i, j, 1] = pos_y
                trans_flag[i, j] = flag

        self._trans = trans_matrix
        self._flag = trans_flag

    def _run(self, sess):
        image = sess.feed_dict.get(self)
        if image is not None:
            return image
        
        if hasattr(self._x, 'run'):
            return self._run_warping(self._x.run(sess))
        return self._run_warping(self._x)

    def _run_warping(self, image):
        ret = np.zeros(shape=(self.shape[0], self.shape[1], 3), dtype=int)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                pos_x, pos_y = self._trans[i, j]
                pos_x = int(pos_x)
                pos_y = int(pos_y)
                if pos_x < 0 or pos_x >= image.shape[0]:
                    continue
                if pos_y < 0 or pos_y >= image.shape[1]:
                    continue
                ret[i, j, :] = image[pos_x, pos_y, :]
                if self._flag[i, j] and self.fix_color:
                    ret[i, j, :] = self.fix_color

        return ret
