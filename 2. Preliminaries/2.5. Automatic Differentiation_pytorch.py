import torch

x = torch.arange(4.0)
# print(x)
x.requires_grad_(True)  # Same as `x = torch.arange(4.0, requires_grad=True)`
# print(x.grad)  # The default value is None
y = 2 * torch.dot(x, x)
# print(y)
y.backward()
# print(x.grad)
# print(x.grad == 4 * x)
# PyTorch accumulates the gradient in default, we need to clear the previous
# values
x.grad.zero_()
y = x.sum()
y.backward()
# print(x.grad)

# Invoking `backward` on a non-scalar requires passing in a `gradient` argument
# which specifies the gradient of the differentiated function w.r.t `self`.
# In our case, we simply want to sum the partial derivatives, so passing
# in a gradient of ones is appropriate
x.grad.zero_()
y = x * x
# y.backward(torch.ones(len(x))) equivalent to the below
y.sum().backward()
# print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
# print(x.grad == u)
x.grad.zero_()
y.sum().backward()
# print(x.grad == 2 * x)

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

print(a.grad == d / a)