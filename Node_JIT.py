import numpy as np
import numba
from Connection_JIT import Connection_JIT

list_instance = numba.typed.List()
list_instance.append(Connection_JIT(0,0,0))

spec = [
    ('id', numba.int64),
    ('bias', numba.f8),
    ('type', numba.types.unicode_type),
    ('connections', numba.typeof(list_instance)),
    ('receptions', numba.typeof(list_instance)),
    ('value', numba.f8),
    ('correction', numba.f8),
    ('affinity', numba.f8),
]

@numba.experimental.jitclass(spec)
class Node_JIT():
    def __init__(self, id, type, _connections, _receptions):
        self.id = id
        if type == "Input":
            self.bias = 0.0
        else:
            self.bias = np.random.randn()
        self.type = type
        self.connections = _connections
        self.connections.pop(0)
        self.receptions = _receptions
        self.receptions.pop(0)
        self.value = 0.0
        self.correction = 0.0
        self.affinity = 0.0

    def to_string(self):
        res = "ID: " + str(self.id) + ", Type: " + self.type + ", Bias: " + self.float_to_string(self.bias) + ", Value: " + self.float_to_string(self.value) + ", Affinity: " + self.float_to_string(self.affinity) + "\n"
        for c in self.connections:
            res = res + "Connected to: " + str(c.output_id) + " with weight: " + self.float_to_string(c.weight) + " and affinity: " + self.float_to_string(c.affinity)
            if c.enabled:
                res = res + " (Enabled), Innov: " + str(c.innov) + "\n"
            else:
                res = res + " (Disabled), Innov: " + str(c.innov) + "\n"
        if len(self.connections) == 0:
            res = res + "No connections\n"
        return res

    def float_to_string(self, value):
        base = 1
        dec = 0
        res = ""
        if value < 0:
            res += "-"
            value *= -1
        digit = value
        while base <= value:
            base = base * 10
            dec += 1
        if dec == 0:
            res += "0."
        else:
            while dec > 0:
                base = np.divide(base, 10)
                res += str(int(np.divide(digit, base)))
                digit = digit%base
                dec -= 1
            res += "."
        digit *= 1000
        base *= 1000
        for i in range(3):
            base = np.divide(base, 10)
            res += str(int(np.divide(digit, base)))
            digit = digit%base
        return res



