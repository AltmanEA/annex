import sys

import numpy as np
from scipy.optimize import curve_fit


def line_approx_fun(x, a, b):
    return a + np.dot(b, x)


def square_approx_fun(x, a, b, c):
    return a + np.dot(b, x) + np.dot(c, np.multiply(x, x))


def cube_approx_fun(x, a, b, c, d):
    return a + np.dot(b, x) + np.dot(c, np.multiply(x, x)) + np.dot(d, np.multiply(np.multiply(x, x), x))


class PiecewiseApprox:

    def __init__(self, x, y, fun, _borders):
        self.x = x
        self.y = y
        self.n = x.size
        self.fun = fun
        self.dim = fun.__code__.co_argcount - 1
        self.borders = _borders

    def piecewise_approx(self):
        _factors = np.zeros((self.borders.size + 1, self.dim))
        start = 0
        bb = np.append(self.borders, self.n)
        for index, border in enumerate(bb):
            popt = curve_fit(self.fun, self.x[start:border], self.y[start:border])[0]
            _factors[index] = popt
            start = border
        self.found_factors = _factors
        return

    def one_border_force(self):
        min_points = self.dim
        _cost = sys.float_info.max
        for i in range(min_points, self.n - min_points + 1):
            self.borders = np.array([i])
            self.piecewise_approx()
            if self.result_cost() < _cost:
                _cost = self.result_cost()
                _borders = self.borders
        self.borders = _borders
        self.piecewise_approx()
        return

    def two_border_force(self):
        min_points = self.dim
        _cost = sys.float_info.max
        for i in range(min_points, self.n - min_points + 1):
            for j in range(min_points + i, self.n - min_points + 1):
                self.borders = np.array([i, j])
                self.piecewise_approx()
                if self.result_cost() < _cost:
                    _cost = self.result_cost()
                    _borders = self.borders
        self.borders = _borders
        self.piecewise_approx()
        return

    def border_force(self, n_border):
        min_points = self.dim
        _cost = sys.float_info.max
        if self.borders.size == 0:
            start = min_points
        else:
            start = self.borders[-1] + min_points
        unsolved_borders = n_border - self.borders.size
        for i in range(start, self.n - unsolved_borders * min_points + 1):
            pa = PiecewiseApprox(self.x, self.y, self.fun, np.append(self.borders, np.array([i])))
            if unsolved_borders > 1:
                pa.border_force(n_border)
            pa.piecewise_approx()
            if pa.result_cost() < _cost:
                _cost = pa.result_cost()
                _borders = pa.borders
        self.borders = _borders
        self.piecewise_approx()
        return

    def result_approx_fun(self):
        start = 0
        result = np.zeros([self.n])
        bb = np.append(self.borders, self.n)
        for index, border in enumerate(bb):
            result[start:border] = self.fun(self.x[start:border], *self.found_factors[index])
            start = border
        return result

    def result_residuals(self):
        return self.result_approx_fun() - self.y

    def result_cost(self):
        _residuals = self.result_residuals()
        return np.sqrt(np.sum(np.multiply(_residuals, _residuals)))


# TEST
def pretty_print(x: np.array):
    print(', '.join(["{0:0.3f}".format(i) for i in x]))


xdata = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
ydata = np.array([0, 1, 4, 9, 16, 4, 4, 4, 4, 4])
borders = np.array([], dtype=int)
pretty_print(ydata)

pa = PiecewiseApprox(xdata, ydata, line_approx_fun, borders)
pa.one_border_force()
print(pa.borders)
print(pa.result_approx_fun())
print(pa.result_cost())
pa = PiecewiseApprox(xdata, ydata, square_approx_fun, borders)
pa.one_border_force()
print(pa.borders)
print(pa.result_approx_fun())
print(pa.result_cost())
pa = PiecewiseApprox(xdata, ydata, line_approx_fun, borders)
pa.two_border_force()
print(pa.borders)
print(pa.result_approx_fun())
print(pa.result_cost())
pa = PiecewiseApprox(xdata, ydata, square_approx_fun, borders)
pa.two_border_force()
print(pa.borders)
print(pa.result_approx_fun())
print(pa.result_cost())
pa = PiecewiseApprox(xdata, ydata, line_approx_fun, borders)
pa.border_force(2)
print(pa.borders)
print(pa.result_approx_fun())
print(pa.result_cost())
pa = PiecewiseApprox(xdata, ydata, square_approx_fun, borders)
pa.border_force(2)
print(pa.borders)
print(pa.result_approx_fun())
print(pa.result_cost())
