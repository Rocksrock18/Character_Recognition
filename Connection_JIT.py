import numpy as np
import numba

spec = [
    ('input_id', numba.int64),
    ('output_id', numba.int64),
    ('weight', numba.f8),
    ('affinity', numba.f8),
    ('enabled', numba.b1),
    ('innov', numba.int64),
]

@numba.experimental.jitclass(spec)
class Connection_JIT():
    def __init__(self, input, output, innov):
        self.input_id = input
        self.output_id = output
        self.weight = np.random.randn()
        self.affinity = 0.0
        self.enabled = True
        self.innov = innov


    def f_to_string(self, value):
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

    def foo(self):
        sum = 0
        for i in range(100):
            for k in range(100):
                for j in range(100):
                    sum += j
    
    def to_string(self):
        res = "Input: " + str(self.input_id) + ", Output: " + str(self.output_id) + ", Weight: " + self.f_to_string(self.weight) + ", Affinity: " + self.f_to_string(self.affinity) + ", Enabled: "
        if self.enabled:
            res = res + "True, Innov: " + str(self.innov)
        else:
            res = res + "False, Innov: " + str(self.innov)
        return res

