import math

import numpy as np


def roots_from_companion(coeffs):
    """
    Solve polynomial by finding eigenvalues of companion matrix.
    Input: coeffs = [a0, a1, ..., an] for a0*x^n + a1*x^{n-1} + ... + an
    Output: array of complex roots
    """
    coeffs = np.trim_zeros(coeffs, "f")  # Remove leading zeros

    n = len(coeffs) - 1
    if n < 1:
        return np.array([])  # Constant or empty polynomial

    # Normalize to make monic (leading coefficient = 1)
    coeffs = coeffs / coeffs[0]

    # Companion matrix
    C = np.zeros((n, n))
    C[1:, :-1] = np.eye(n - 1)
    C[0, :] = -coeffs[1:]

    # Compute eigenvalues
    roots = np.linalg.eigvals(C)
    return roots


def max_radius(a, b, c):
    if c == 0.0:
        discrim = a * a - 4 * b
        if np.isfinite(discrim) and discrim >= 0.0:
            discrim = np.sqrt(discrim) - a
            if discrim > 0.0:
                return 2.0 / discrim
    else:
        # commonly used terms
        boc = b / c
        boc2 = boc * boc
        t1 = (9 * a * boc - 2 * b * boc2 - 27) / c
        t2 = 3 * a / c - boc2
        discrim = t1 * t1 + 4 * t2 * t2 * t2
        if discrim >= 0.0:
            d2 = np.sqrt(discrim)
            discrim = np.cbrt((np.sqrt(discrim) + t1) / 2.0)
            soln = (discrim - (t2 / discrim) - boc) / 3.0
            if soln > 0.0:
                return soln
        else:
            theta = (math.atan2(np.sqrt(-discrim), t1)) / 3.0
            twothirdpi = 2.0 * math.pi / 3.0
            # by construction, if discrim < 0 then t2 < 0, so the sqrt is safe
            t3 = 2.0 * np.sqrt(-t2)
            solns = [
                (t3 * np.cos(theta + i * twothirdpi) - boc) / 3 for i in [-1, 0, 1]
            ]
            pos_solns = [s for s in solns if np.isfinite(s) and s > 0.0]
            if pos_solns:
                return np.min(pos_solns)
    return np.inf


def eval_poly_horner(poly, x):
    # Evaluates a polynomial y=f(x) with
    #
    # f(x) = c_0*x^0 + c_1*x^1 + c_2*x^2 + c_3*x^3 + c_4*x^4 ...
    #
    # given by poly_coefficients c_i at points x using numerically stable
    # Horner scheme.
    #
    # The degree of the polynomial is N_COEFFS - 1

    y = 0.0
    for coeff in reversed(poly):
        y = x * y + coeff
    return y


def eval_poly_odd_horner(poly_odd, x):
    # Evaluates an odd-only polynomial y=f(x) with
    #
    # f(x) = c_0*x^1 + c_1*x^3 + c_2*x^5 + c_3*x^7 + c_4*x^9 ...
    #
    # given by poly_coefficients c_i at points x using numerically stable
    # Horner scheme.
    #
    # The degree of the polynomial is 2*N_COEFFS - 1

    return x * eval_poly_horner(
        poly_odd, x * x
    )  # evaluate x^2-based "regular" polynomial after facting out one x term


def eval_poly_even_horner(poly_even, x):
    # Evaluates an even-only polynomial y=f(x) with
    #
    # f(x) = c_0 + c_1*x^2 + c_2*x^4 + c_3*x^6 + c_4*x^8 ...
    #
    # given by poly_coefficients c_i at points x using numerically stable
    # Horner scheme.
    #
    # The degree of the polynomial is 2*(N_COEFFS - 1)

    return eval_poly_horner(
        poly_even, x * x
    )  # evaluate x^2-substituted "regular" polynomial


class PolynomialProxy:
    def __init__(self, coeffs, type):
        self.coeffs = coeffs
        self.type = type

    def eval_horner(self, x):
        if self.type == "FULL":
            # Evaluate a full polynomial
            return eval_poly_horner(self.coeffs, x)
        elif self.type == "EVEN":
            # Evaluate an even-only polynomial
            return eval_poly_even_horner(self.coeffs, x)
        elif self.type == "ODD":
            # Evaluate an odd-only polynomial
            return eval_poly_odd_horner(self.coeffs, x)
        else:
            raise ValueError("Invalid polynomial type")


def eval_poly_inverse_horner_newton(
    poly: PolynomialProxy,
    dpoly: PolynomialProxy,
    inv_poly_approx: PolynomialProxy,
    y: float,
    n_newton_iterations: int,
):
    # Evaluates the inverse x = f^{-1}(y) of a reference polynomial y=f(x)
    # (given by poly_coefficients) at points y using numerically stable Horner
    # scheme and Newton iterations starting from an approximate solution
    # \\hat{x} = \\hat{f}^{-1}(y) (given by inv_poly_approx) and the
    # polynomials derivative df/dx (given by poly_derivative_coefficients)

    assert n_newton_iterations >= 0, "Require at least a single Newton iteration"

    # approximation / starting points - also returned for zero iterations
    x = inv_poly_approx.eval_horner(y)

    converged = False
    for _ in range(n_newton_iterations):
        dfdx = dpoly.eval_horner(x)
        if dfdx == 0.0:
            break
        residual = poly.eval_horner(x) - y
        dx = residual / dfdx
        x -= dx
        if abs(dx) < 1e-6:
            converged = True
            break

    return x, converged


success = 0

for _ in range(1000):
    k1, k2, k3, k4 = (np.random.rand(4) - 0.5) * 2.0
    k4 = 0.0

    poly = PolynomialProxy([1.0, 3 * k1, 5 * k2, 7 * k3, 9 * k4], "EVEN")
    dpoly = PolynomialProxy([6 * k1, 20 * k2, 42 * k3, 72 * k4], "ODD")

    x0 = np.sqrt(max_radius(3 * k1, 5 * k2, 0))
    if x0 == np.inf:
        x0 = 3.14 / 2
    inv_poly_approx = PolynomialProxy([x0], "FULL")

    # inv_poly_approx = PolynomialProxy([3.14/2], 'FULL')

    x, converged = eval_poly_inverse_horner_newton(
        poly, dpoly, inv_poly_approx, 0.0, 100
    )
    _x = np.sqrt(max_radius(3 * k1, 5 * k2, 7 * k3))
    if not converged and _x == np.inf:
        success += 1
    elif converged and abs(x - _x) < 1e-6:
        success += 1
    else:
        print(x, converged, _x)
        roots = np.sqrt(roots_from_companion([9 * k4, 7 * k3, 5 * k2, 3 * k1, 1.0]))
        print(roots)

print(success / 1000.0)


# print (poly.eval_horner(x))
# print (x, converged)

# a = poly.coeffs[1]
# b = poly.coeffs[2]
# delta = a*a - 4*b
# sol1 = (-a - delta**0.5) / (2*b)
# sol2 = (-a + delta**0.5) / (2*b)
# print (sol1**0.5, sol2**0.5)


# def solve_fisheye_derivative(k1, k2, k3, k4):
#     import sympy

#     # Define the symbolic variable
#     x = sympy.Symbol('x', real=True)

#     # Define the quartic polynomial in x = theta^2
#     p = 9*k4*x**4 + 7*k3*x**3 + 5*k2*x**2 + 3*k1*x + 1

#     # Solve p = 0 symbolically
#     solutions = sympy.solve(sympy.Eq(p, 0), x, dict=True)

#     # Filter out the real, nonnegative solutions (x >= 0)
#     real_solutions = []
#     for sol in solutions:
#         val = sol[x]
#         # Check that val is real and >= 0
#         if val.is_real and val >= 0:
#             real_solutions.append(val)

#     if not real_solutions:
#         # No real nonnegative solution => derivative doesn't cross zero in a valid range
#         return 1e10

#     # Choose the smallest positive solution for x = theta^2
#     x_min = min(real_solutions)

#     # Return theta = sqrt(x_min)
#     theta_max = sympy.sqrt(x_min)
#     return theta_max

# print (solve_fisheye_derivative(k1, k2, k3, k4))
# # 1.32627397261985
