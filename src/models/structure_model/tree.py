class Tree(object):
    def __init__(self, idx):
        self.idx = idx
        self.parent = None
        self.children = list()
        self.num_children = 0

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        count = 0
        if self.parent is not None:
            self._depth = self.parent.depth() + 1
        else:
            self._depth = count
        return self._depth
