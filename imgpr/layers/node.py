import numpy as np

class node:
    def __init__(self, shape):
        self.shape = shape[:2]
    
    def run(self, sess):
        image = sess.feed_dict.get(self)
        if image is not None:
            assert(image.shape[:2] == self.shape)
            return image
        image = self._run(sess) # pylint: disable=assignment-from-no-return
        sess.feed_dict[self] = image
        return image
        
    def _run(self, sess):
        raise Exception("No Data")

class placeholder(node):
    def __init__(self, shape, fix_color=(0, 0, 0)):
        super().__init__(shape)
        self.fix_color = fix_color

    def _run(self, sess):
        r, g, b = self.fix_color
        image = np.ones(shape=(self.shape[0], self.shape[1], 3), dtype=int)
        image[:, :, 0] = r
        image[:, :, 1] = g
        image[:, :, 2] = b
        return image
