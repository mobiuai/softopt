"""
SoftTensor: the soft algebra made visible on tensors -- and the check that it IS what
torch.func.jvp computes (Appendix A.2: dual numbers = single-axis soft numbers).

A SoftTensor is a pair (a, b) of tensors = a0 _+ b (book notation): b the value, a the soft part.
  sum      (a1, b1) + (a2, b2) = (a1 + a2, b1 + b2)                    Section 3.3.1
  product  (a1, b1) * (a2, b2) = (a1*b2 + a2*b1, b1*b2)                Section 3.3.2
  function f(a0 _+ b)          = a*f'(b) 0 _+ f(b)                     Lemma 6.1
Nesting (a SoftTensor whose parts are SoftTensors) gives two soft axes, each with eps^2 = 0; the
mixed part is the exact second directional derivative D2. SoftOptTorch obtains D1 and D2 from
nested torch.func.jvp; this file shows the two are the same numbers.

For presentation and testing only -- the optimizer keeps PyTorch's engine for speed.
    python3 examples/soft_tensor_demo.py
"""
import torch


class SoftTensor:
    def __init__(self, a, b):
        self.a, self.b = a, b

    @staticmethod
    def lift(x):
        return x if isinstance(x, SoftTensor) else SoftTensor(0.0 * x if torch.is_tensor(x) else 0.0, x)
    # make python numbers / tensors on the left of an operator defer to SoftTensor
    __array_priority__ = 1000

    def __add__(self, o):
        o = SoftTensor.lift(o)
        return SoftTensor(self.a + o.a, self.b + o.b)
    __radd__ = __add__

    def __sub__(self, o):
        o = SoftTensor.lift(o)
        return SoftTensor(self.a - o.a, self.b - o.b)

    def __rsub__(self, o):
        return SoftTensor.lift(o) - self

    def __neg__(self):
        return SoftTensor(-1.0 * self.a, -1.0 * self.b)

    def __mul__(self, o):
        o = SoftTensor.lift(o)
        return SoftTensor(self.a * o.b + o.a * self.b, self.b * o.b)
    __rmul__ = __mul__

    def sum(self):
        return SoftTensor(_sum(self.a), _sum(self.b))

    # Lemma 6.1 with f = sin, cos, exp, tanh
    def sin(self):
        return SoftTensor(self.a * cos(self.b), sin(self.b))

    def cos(self):
        return SoftTensor(-1.0 * self.a * sin(self.b), cos(self.b))

    def exp(self):
        e = exp(self.b)
        return SoftTensor(self.a * e, e)

    def tanh(self):
        t = tanh(self.b)
        return SoftTensor(self.a * (1.0 - t * t), t)


def _sum(x):
    return x.sum() if isinstance(x, SoftTensor) or torch.is_tensor(x) else x


def sin(x):  return x.sin() if isinstance(x, SoftTensor) else torch.sin(x)      # noqa: E704
def cos(x):  return x.cos() if isinstance(x, SoftTensor) else torch.cos(x)      # noqa: E704
def exp(x):  return x.exp() if isinstance(x, SoftTensor) else torch.exp(x)      # noqa: E704
def tanh(x): return x.tanh() if isinstance(x, SoftTensor) else torch.tanh(x)    # noqa: E704


def model(theta, W):
    """A small smooth objective: a two-layer tanh map of the parameters, plus trig terms."""
    h = tanh(theta * W[0] + W[1])
    return (h * h).sum() + (sin(theta) * cos(theta * W[2])).sum() + exp(theta * 0.1).sum()


def soft_D1_D2(theta, delta, W):
    """theta + eps1*delta + eps2*delta as a nested SoftTensor; one pass gives f, D1, D2."""
    x = SoftTensor(SoftTensor(torch.zeros_like(delta), delta), SoftTensor(delta, theta))
    r = model(x, W)
    return r.b.b, r.a.b, r.a.a              # f, D1, D2


def jvp_D1_D2(theta, delta, W):
    """What SoftOptTorch does: nested forward-mode AD."""
    f = lambda t: model(t, W)                                              # noqa: E731
    d1 = lambda t: torch.func.jvp(f, (t,), (delta,))[1]                    # noqa: E731
    D1, D2 = torch.func.jvp(d1, (theta,), (delta,))
    return f(theta), D1, D2


if __name__ == '__main__':
    torch.manual_seed(0)
    worst = 0.0
    for trial in range(5):
        n = 50
        theta = torch.randn(n, dtype=torch.float64)
        delta = torch.randint(0, 2, (n,), dtype=torch.float64) * 2 - 1
        W = torch.randn(3, n, dtype=torch.float64)
        s, j = soft_D1_D2(theta, delta, W), jvp_D1_D2(theta, delta, W)
        err = max(abs(float(x - y)) / max(1.0, abs(float(y))) for x, y in zip(s, j))
        worst = max(worst, err)
        print(f'trial {trial}:  SoftTensor  f={float(s[0]):+.12f}  D1={float(s[1]):+.12f}  D2={float(s[2]):+.12f}')
        print(f'           torch jvp   f={float(j[0]):+.12f}  D1={float(j[1]):+.12f}  D2={float(j[2]):+.12f}')
    print(f'\nlargest relative difference: {worst:.1e}  ->  {"identical (to rounding)" if worst < 1e-12 else "MISMATCH"}')
