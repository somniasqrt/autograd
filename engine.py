from __future__ import annotations
import math

class Value:
    def backward(self: Value):
        topo = []
        visited = set()

        def build_topo(node: Value):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)
                topo.append(node)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
        
        
    def __init__(self, data, op='', _prev=()):
        self.data = data
        self.grad = 0.0
        self.op = op
        self._prev = _prev
        self._backward = lambda: None

    def __add__ (self, other: Value):
        out = Value(self.data + other.data, '+', (self,other))
        def _backward():
            self.grad += out.grad * 1
            other.grad += out.grad * 1
        out._backward = _backward
        return out
            
    def __sub__ (self, other: Value):
        out = Value(self.data - other.data, '-', (self,other))
        def _backward():
            self.grad += out.grad * 1
            other.grad += out.grad * -1
        out._backward = _backward
        return out

    def __mul__ (self, other: Value):
        out = Value(self.data * other.data, '*', (self,other))
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __truediv__ (self, other: Value):
        out = Value(self.data / other.data, '/', (self,other))
        def _backward():
            self.grad += out.grad * (1/other.data)
            other.grad += out.grad * (-self.data * 1/other.data**2)
        out._backward = _backward
        return out

    def __pow__ (self, other: Value | float):
        if isinstance(other, Value):
            out = Value(self.data ** other.data, '**', (self,other))
            def _backward():
                self.grad += out.grad * other.data * (self.data ** (other.data - 1))
                other.grad += out.grad * (self.data ** other.data) * math.log(self.data)
            out._backward = _backward
        else:
            out = Value(self.data ** other, '**', (self,))
            def _backward():
                self.grad += out.grad * other * (self.data ** (other - 1))
            out._backward = _backward
        return out
        
    def __neg__(self):
        out = Value(self.data * -1, '-u', (self,))
        def _backward():
            self.grad += out.grad * -1
        out._backward = _backward
        return out