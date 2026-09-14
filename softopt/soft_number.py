"""
Soft-number primitives, cited exactly to *Foundations of Soft Logic*
(Klein & Maimon, Springer 2024, Ch. 3 and Ch. 6). Two ways to use them:

1. Tuple functions (`sadd`, `smul`, ...) -- what every validated SoftOpt
   domain (VQE, GRAPE, camera calibration, portfolio, ...) actually uses
   internally, since they run inside tight optimization loops.
2. `SoftNumber`, a first-class object wrapping the same functions with
   operator overloading, for readability when performance isn't critical:

    from softopt import SoftNumber
    x = SoftNumber(2.0, 3.0)   # 2.0 bridge 3 (book notation, Ch. 3.3)
    y = SoftNumber(1.0, 4.0)
    x + y, x * y, x / y, x ** 3, x.sqrt()

See README.md for the full operation-to-book-source citation table.
"""
import numpy as np


# --- Section 3.3.1: BN is an abelian group under addition ---
def sadd(x, y):
    return (x[0] + y[0], x[1] + y[1])


def sneg(x):
    return (-x[0], -x[1])


def ssub(x, y):
    return sadd(x, sneg(y))


# --- Section 3.3.2: BN is a ring under addition and multiplication ---
def smul(x, y):
    a1, b1 = x
    a2, b2 = y
    return (a1 * b2 + a2 * b1, b1 * b2)


# --- Section 3.3.3: BN is "almost a field" -- Lemma 3.1 (inverse) ---
def sinv(x):
    a, b = x
    if b == 0:
        # Lemma 3.2: numbers on the zero axis (b=0) have no inverse
        raise ZeroDivisionError(
            "a soft number with b=0 lies on the zero axis and has no "
            "inverse (Lemma 3.2, Foundations of Soft Logic)"
        )
    return (-a / b**2, 1.0 / b)


def sdiv(x, y):
    return smul(x, sinv(y))


# --- Section 3.3.4: Lemma 3.3, integer powers ---
def spow(x, n):
    """Integer powers via Lemma 3.3; fractional powers by composing
    Lemma 3.3 with Lemma 3.5's root (e.g. x**2.5 = (x**5)**(1/2))."""
    a, b = x
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if isinstance(n, int):
        if n < 0:
            return sinv(spow(x, -n))
        if n == 0:
            return (0.0, 1.0)
        return (n * a * b ** (n - 1), b**n)
    # fractional: x**(p/q). Use the chain rule form directly -- for
    # f(t)=t^n the extension rule gives (a*n*b^(n-1), b^n), which holds
    # for real n as well, by Lemma 6.1's general f(a,b)=(a*f'(b), f(b)).
    return (a * n * b ** (n - 1), b**n)


# --- Section 3.3.5: Lemma 3.4 (square root), Lemma 3.5 (n-th root) ---
def sroot(x, n):
    """n-th root, positive branch. Derived by inverting Lemma 3.3's power
    formula; reduces exactly to Lemma 3.4's closed form for n=2. Undefined
    for b=0 (Lemma 3.4/3.5's b=0 case)."""
    a, b = x
    if b == 0:
        raise ZeroDivisionError(
            "root of a soft number with b=0 is undefined "
            "(Lemma 3.4/3.5, Foundations of Soft Logic)"
        )
    if b < 0 and n % 2 == 0:
        raise ValueError(f"even root (n={n}) of a soft number with b<0 "
                          "is not real-valued (Lemma 3.4)")
    if b < 0:
        # Odd root of a negative real component: Python's ** returns a complex
        # principal root here, so take the real root of |b| and restore the sign.
        b_root = -((-b) ** (1.0 / n))
    else:
        b_root = b ** (1.0 / n)
    # d/db b^(1/n) = b^(1/n) / (n*b), valid for either sign
    a_root = a * b_root / (n * b)
    return (a_root, b_root)


def ssqrt(x):
    """Lemma 3.4, positive branch."""
    return sroot(x, 2)


# --- elementary-function extension: Lemma 6.1 (p.40) / Section 6.2 (p.41) ---
def ssin(x):
    a, b = x
    return (a * np.cos(b), np.sin(b))


def scos(x):
    a, b = x
    return (-a * np.sin(b), np.cos(b))


def sexp(x):
    a, b = x
    e = np.exp(b)
    return (a * e, e)


def slog(x):
    """Natural logarithm, Lemma 6.1's general rule with f(t)=ln(t)."""
    a, b = x
    if b <= 0:
        raise ValueError("slog requires a positive real component")
    return (a / b, np.log(b))


class SoftNumber:
    """Object wrapper around the tuple functions above -- same math,
    operator syntax. See module docstring."""
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a = a
        self.b = b

    @staticmethod
    def _coerce(other):
        if isinstance(other, SoftNumber):
            return other
        return SoftNumber(0.0, other)  # a real number embeds as (0, b)

    def as_tuple(self):
        return (self.a, self.b)

    @classmethod
    def _from_tuple(cls, t):
        return cls(t[0], t[1])

    def __add__(self, other):
        return self._from_tuple(sadd(self.as_tuple(), self._coerce(other).as_tuple()))

    __radd__ = __add__

    def __neg__(self):
        return self._from_tuple(sneg(self.as_tuple()))

    def __sub__(self, other):
        return self._from_tuple(ssub(self.as_tuple(), self._coerce(other).as_tuple()))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        return self._from_tuple(smul(self.as_tuple(), self._coerce(other).as_tuple()))

    __rmul__ = __mul__

    def inverse(self):
        return self._from_tuple(sinv(self.as_tuple()))

    def __truediv__(self, other):
        return self._from_tuple(sdiv(self.as_tuple(), self._coerce(other).as_tuple()))

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __pow__(self, n):
        if isinstance(n, SoftNumber):
            # x**y with both soft: x**y = exp(y * ln x)
            return (self.log() * n).exp()
        return self._from_tuple(spow(self.as_tuple(), n))

    def __rpow__(self, base):
        """base ** self, where base is an ordinary number.
        d/dt base^t = base^t * ln(base)."""
        if base <= 0:
            raise ValueError(f"{base}**SoftNumber requires a positive base")
        val = base ** self.b
        return SoftNumber(self.a * val * np.log(base), val)

    def log(self):
        """Natural log, via Lemma 6.1's general rule with f=ln."""
        return self._from_tuple(slog(self.as_tuple()))

    def tanh(self):
        t = np.tanh(self.b)
        return SoftNumber(self.a * (1.0 - t * t), t)

    def log10(self):
        return SoftNumber(self.a / (self.b * np.log(10.0)), np.log10(self.b))

    def __abs__(self):
        """|x|. The real component picks the branch. At exactly zero the
        derivative of |x| does not exist, so the zero-axis component is
        reported as nan rather than silently taking one side."""
        if self.b < 0:
            return -self
        if self.b == 0:
            return SoftNumber(float("nan") if self.a != 0 else 0.0, 0.0)
        return SoftNumber(self.a, self.b)

    # --- comparisons. These order by the REAL component only, which is what
    # lets a model branch on a parameter value; the compiled derivative is then
    # valid on whichever branch the parameters fall in. Note this is a practical
    # ordering for tracing, not the book's Definition 3.1 ordering of the zero
    # axis itself (a0 < b0 when a < b), which concerns the other component.
    def __lt__(self, other):
        return self.b < self._coerce(other).b

    def __le__(self, other):
        return self.b <= self._coerce(other).b

    def __gt__(self, other):
        return self.b > self._coerce(other).b

    def __ge__(self, other):
        return self.b >= self._coerce(other).b

    def __float__(self):
        """The real component. Note this DISCARDS the derivative -- if your
        model calls float() on a traced value, the compiled derivative will
        be wrong, which soft_compile's verification is designed to catch."""
        return float(self.b)

    def root(self, n):
        return self._from_tuple(sroot(self.as_tuple(), n))

    def sqrt(self):
        return self._from_tuple(ssqrt(self.as_tuple()))

    def sin(self):
        return self._from_tuple(ssin(self.as_tuple()))

    def cos(self):
        return self._from_tuple(scos(self.as_tuple()))

    def exp(self):
        return self._from_tuple(sexp(self.as_tuple()))

    def __eq__(self, other):
        other = self._coerce(other)
        return self.a == other.a and self.b == other.b

    def __repr__(self):
        return f"SoftNumber({self.a!r}, {self.b!r})"
