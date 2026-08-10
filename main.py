from engine import Value
if __name__ == "__main__":
    x = Value(2.0)
    y = Value(5.0)
    L = (x / y) * (y**x) 
    L.backward()
    print(x.grad)
    print(y.grad)
    