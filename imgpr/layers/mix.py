import numpy as np

from .node import node

class mix(node):
    def __init__(self, x1, x2, alpha):
        super().__init__(x1.shape)

        assert(x1.shape[:2] == x2.shape[:2])

        self._x1 = x1
        self._x2 = x2
        if type(alpha) in [int, float]:
            self._alpha = lambda x, y : alpha
        else:
            self._alpha = alpha

    def _run(self, sess):
        image = sess.feed_dict.get(self)
        if image is not None:
            return image
        
        if hasattr(self._x1, 'run'):
            x1 = self._x1.run(sess)
        else:
            x1 = self._x1

        if hasattr(self._x2, 'run'):
            x2 = self._x2.run(sess)
        else:
            x2 = self._x2

        return self._run_mix(x1, x2)

    def _run_mix(self, image, target):
        alpha = np.ones_like(image, dtype=float)
        for i in range(alpha.shape[0]):
            for j in range(alpha.shape[1]):
                alpha[i, j, :] *= self._alpha(i / (alpha.shape[0] - 1), j / (alpha.shape[1] - 1))
        return (image * (1 - alpha) + target * alpha).astype(int)
