from collections.abc import Iterable

class Session:
    def __init__(self):
        self.feed_dict = {}
        self.nodes = {}

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        pass

    def _run_single(self, node):
        if hasattr(node, 'run'):
            image = node.run(self)
        else:
            image = node
        return image

    def run(self, nodes, feed_dict={}):
        self.feed_dict.update(feed_dict)
        
        if isinstance(nodes, Iterable):
            ret = []
            for node in nodes:
                ret = ret + [self._run_single(node)]
            ret = tuple(ret)
        else:
            ret = self._run_single(nodes)

        self.feed_dict.clear()
        return ret
