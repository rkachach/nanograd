from nanograd.nn import MLP
from nanograd.value import Value

n = MLP(3, [4,4, 1])

# input parameters
xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0],]

# desired targets
ys = [1.0, -1.0, -1.0, 1.0]

# training loop
for k in range(1000):

    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # zero grad all the network parameters
    for p in n.parameters():
        p.grad = 0

    # backward pass
    loss.backward()

    # update the gradient
    for p in n.parameters():
        p.data += -0.05 * p.grad

print(f"loss value: {loss.data}")
print(f"predicted values: {ypred}")
