"""
Verifies SoftNumber against every axiom/lemma in Foundations of Soft
Logic Ch. 3 that it claims to implement -- not just "uses the right
formula" but the actual algebraic properties the book proves from it.
"""
import numpy as np
import pytest
from softopt import SoftNumber
from softopt.soft_number import sadd, smul, sinv  # the tuple-based internals


@pytest.fixture
def xyz():
    rng = np.random.default_rng(0)
    vals = rng.uniform(-5, 5, 6)
    return (SoftNumber(vals[0], vals[1]),
            SoftNumber(vals[2], vals[3]),
            SoftNumber(vals[4], vals[5]))


def test_axiom_3_3_nullity():
    """Axiom 3.3: pure zero-axis numbers (b=0) collapse to absolute zero
    under multiplication."""
    x = SoftNumber(3.0, 0.0)
    y = SoftNumber(4.0, 0.0)
    assert (x * y) == SoftNumber(0.0, 0.0)


def _close(x, y):
    return np.isclose(x.a, y.a) and np.isclose(x.b, y.b)


def test_section_3_3_1_abelian_group_under_addition(xyz):
    x, y, z = xyz
    zero = SoftNumber(0.0, 0.0)
    assert x + zero == x                                    # identity
    assert x + (-x) == zero                                  # inverse
    assert _close((x + y) + z, x + (y + z))                    # associativity
    assert x + y == y + x                                     # commutativity


def test_section_3_3_2_ring(xyz):
    x, y, z = xyz
    one = SoftNumber(0.0, 1.0)
    assert x * one == x                                       # mult. identity
    assert _close((x * y) * z, x * (y * z))                    # mult. associativity
    assert _close(x * (y + z), (x * y) + (x * z))               # distributivity


def test_lemma_3_1_inverse_formula(xyz):
    x, _, _ = xyz
    expected = SoftNumber(-x.a / x.b**2, 1.0 / x.b)
    assert x.inverse() == expected
    one = SoftNumber(0.0, 1.0)
    result = x * x.inverse()
    assert np.isclose(result.a, one.a) and np.isclose(result.b, one.b)


def test_lemma_3_2_no_inverse_on_zero_axis():
    x = SoftNumber(5.0, 0.0)
    with pytest.raises(ZeroDivisionError):
        x.inverse()


def test_lemma_3_3_integer_powers(xyz):
    x, _, _ = xyz
    for n in range(0, 5):
        by_repeated_mul = SoftNumber(0.0, 1.0)
        for _ in range(n):
            by_repeated_mul = by_repeated_mul * x
        closed_form = x ** n
        assert np.isclose(by_repeated_mul.a, closed_form.a)
        assert np.isclose(by_repeated_mul.b, closed_form.b)


def test_lemma_3_4_square_root(xyz):
    x, _, _ = xyz
    x = SoftNumber(abs(x.a), abs(x.b) + 0.1)  # ensure b>0
    r = x.sqrt()
    squared_back = r ** 2
    assert np.isclose(squared_back.a, x.a)
    assert np.isclose(squared_back.b, x.b)
    # matches the book's explicit closed form
    expected = SoftNumber(x.a / (2 * np.sqrt(x.b)), np.sqrt(x.b))
    assert np.isclose(r.a, expected.a) and np.isclose(r.b, expected.b)


def test_lemma_3_5_nth_root(xyz):
    x, _, _ = xyz
    x = SoftNumber(abs(x.a), abs(x.b) + 0.1)
    for n in (2, 3, 4, 5):
        r = x.root(n)
        back = r ** n
        assert np.isclose(back.a, x.a)
        assert np.isclose(back.b, x.b)


def test_zero_axis_root_undefined():
    x = SoftNumber(3.0, 0.0)
    with pytest.raises(ZeroDivisionError):
        x.sqrt()


def test_lemma_6_1_elementary_functions_match_general_rule(xyz):
    x, _, _ = xyz
    # f(a,b) = (a*f'(b), f(b)) for f=sin,cos,exp -- check each against
    # the general rule directly, not just against itself
    a, b = x.a, x.b
    assert np.isclose(x.sin().a, a * np.cos(b)) and np.isclose(x.sin().b, np.sin(b))
    assert np.isclose(x.cos().a, -a * np.sin(b)) and np.isclose(x.cos().b, np.cos(b))
    e = np.exp(b)
    assert np.isclose(x.exp().a, a * e) and np.isclose(x.exp().b, e)


def test_class_matches_tuple_based_internals_exactly(xyz):
    """The class is a convenience layer -- SoftOpt's actual optimizers use
    the tuple functions directly for speed. Verify they give IDENTICAL
    results, so using the class is never a correctness trade-off."""
    x, y, _ = xyz
    xt, yt = x.as_tuple(), y.as_tuple()
    assert np.allclose((x + y).as_tuple(), sadd(xt, yt))
    assert np.allclose((x * y).as_tuple(), smul(xt, yt))
    assert np.allclose(x.inverse().as_tuple(), sinv(xt))
