import numpy as np

class Node():
    def __init__(self, id, type):
        self.id = id
        self.connections = []
        self.bias = (np.random.randn(), 0) [type == "Input"] # input nodes has no bias
        self.type = type
        self.value = 0

    def __str__(self):
        res = "ID: " + str(self.id) + ", Type: " + self.type + ", Bias: " + str(self.bias) + ", Value: " + str(self.value) + "\n"
        if(len(self.connections) == 0):
            res += "No connections\n"
        for c in self.connections:
            res += "Connected to: " + str(c.output_id) + " with weight: " + str(c.weight) + (" (Disabled)", " (Enabled)") [c.enabled] + ", Innov: " + str(c.innov) + "\n"
        return res


