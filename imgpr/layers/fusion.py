import numpy as np

from .node import node

from imgpr.fusion import FusionModel

class fusion(node):
    def __init__(self, image, model):
        super().__init__(image.shape)

        assert(isinstance(model, FusionModel))
        assert(image.shape[:2] == model.shape)

        self._image = image
        self._model = model

    def _run(self, sess):
        image = sess.feed_dict.get(self)
        if image is not None:
            return image
        
        if hasattr(self._image, 'run'):
            image = self._image.run(sess)
        else:
            image = self._image

        return self._model._run(sess, image)
