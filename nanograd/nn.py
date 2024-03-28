import random
from nanograd.value import Value

class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1), label='w') for _ in range(nin)]
        self.b = Value(random.uniform(-1,1), label='b')

    def __call__(self, x):
        act = sum ((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self. neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = [p for neuron in self.neurons for p in neuron.parameters()]
        return params

class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz [i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = [p for layer in self.layers for p in layer.parameters()]
        return params
