from imgpr.layers.node import node

class FusionModel(node):
    def __init__(self, source, mask):
        assert(source.shape[:2] == mask.shape)
        self.shape = source.shape[:2]
        self._source = source
        self._mask = mask
        self._inited = False

    def _run(self, sess, target):
        if not self._inited:
            if hasattr(self._source, 'run'):
                source = self._source.run(sess)
            else:
                source = self._source
            if hasattr(self._mask, 'run'):
                mask = self._mask.run(sess)
            else:
                mask = self._mask
            self._init(source, mask)
            self._inited = True

        return self._run_fusion(target).astype(int)

    def _init(self, source, mask):
        raise "Not implemented"

    def _run_fusion(self, image):
        raise "Not implemented"
