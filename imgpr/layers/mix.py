import numpy as np

from .node import node

class mix(node):
    def __init__(self, x1, x2, alpha):
        super().__init__(x1.shape)

        assert(x1.shape == x2.shape)
        assert(0 <= alpha <= 1)

        self._x1 = x1
        self._x2 = x2
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
        return (image * (1 - self._alpha) + target * self._alpha).astype(int)
