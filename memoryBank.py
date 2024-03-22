import torch
from collections import deque


class MemoryBank:
    def __init__(self, size=None):
        """
        :param size: Maximum number of features to be stored. If None, it grows indefinitely.
        """
        self.memory = deque(maxlen=size)
        self.length = 0
        self.is_empty = 1

    def add(self, feature):
        """
        Adds a feature to the memory bank.
        :param feature: The feature to be added, typically a tensor.
        """
        self.memory.append(feature)
        self.length += 1
        self.is_empty = 0

    def get_memory(self):
        """
        :return: All the features in the memory bank.
        """
        return torch.stack(list(self.memory), dim=0)
    
    def empty(self):
        return self.is_empty
    
    def pop(self):
        self.length -= 1
        self.memory.popleft()

    def get_length(self):
        return self.length