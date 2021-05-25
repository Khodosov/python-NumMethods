import numpy as np
from copy import copy
from scipy.integrate import RK45, solve_ivp
from scipy.optimize import fsolve

import NumMethods_S3T2.coeffs_collection as collection
from utils.ode_collection import ODE


class OneStepMethod:
    def __init__(self, **kwargs):
        self.name = 'default_method'
        self.p = None  # порядок
        self.__dict__.update(**kwargs)

    def step(self, ode: ODE, t, y, dt):
        """
        делаем шаг: t => t+dt, используя ode(t, y)
        """
        raise t+dt


class ExplicitEulerMethod(OneStepMethod):
    """
    Явный метод Эйлера (ничего менять не нужно)
    """
    def __init__(self):
        super().__init__(name='Euler (explicit)', p=1)

    def step(self, ode: ODE, t, y, dt):
        return y + dt * ode(t, y)


class ImplicitEulerMethod(OneStepMethod):
    """
    Неявный метод Эйлера
    Подробности: https://en.wikipedia.org/wiki/Backward_Euler_method
    """
    def __init__(self):
        super().__init__(name='Euler (implicit)', p=1)

    def step(self, ode: ODE, t, y, dt):
        def leftside(yn1):
            return yn1 - dt * ode(t, yn1) - y
        return fsolve(leftside, y)


class RungeKuttaMethod(OneStepMethod):
    """
    Явный метод Рунге-Кутты с коэффициентами (A, b)
    Замените метод step() так, чтобы он не использовал встроенный класс RK45
    """
    def __init__(self, coeffs: collection.RKScheme):
        super().__init__(**coeffs.__dict__)

    def step(self, ode: ODE, t, y, dt):
        A, b = self.A, self.b
        c = np.sum(A, axis=1)
        k = []
        for i in range(len(A)):
            s = 0
            for j in range(i):
                s += A[i, j] * k[j]
            k.append(ode(t + dt * c, y + dt * s))
        K = np.array(k)
        return y + dt * (b @ K)


class EmbeddedRungeKuttaMethod(RungeKuttaMethod):
    """
    Вложенная схема Рунге-Кутты с параметрами (A, b, e):
    """
    def __init__(self, coeffs: collection.EmbeddedRKScheme):
        super().__init__(coeffs=coeffs)

    def embedded_step(self, ode: ODE, t, y, dt):
        """
        Шаг с использованием вложенных методов:
        y1 = RK(ode, A, b)
        y2 = RK(ode, A, b+e)

        :return: приближение на шаге (y1), разность двух приближений (dy = y2-y1)
        """
        A, b, e = self.A, self.b, self.e
        c = np.sum(A, axis=1)
        k = []
        for i in range(len(A)):
            s = 0
            for j in range(i):
                s += A[i, j] * k[j]
            k.append(ode(t + dt * c, y + dt * s))
        K = np.array(k)
        return y + dt * (b @ K), e @ K
        # return y1, dy


class EmbeddedRosenbrockMethod(OneStepMethod):
    """
    Вложенный метод Розенброка с параметрами (A, G, gamma, b, e)
    Подробности: https://dl.acm.org/doi/10.1145/355993.355994 (уравнение 2)
    """
    def __init__(self, coeffs: collection.EmbeddedRosenbrockScheme):
        super().__init__(**coeffs.__dict__)

    def embedded_step(self, ode: ODE, t, y, dt):
        """
        Шаг с использованием вложенных методов:
        y1 = Rosenbrock(ode, A, G, gamma, b)
        y2 = Rosenbrock(ode, A, G, gamma, b+e)

        :return: приближение на шаге (y1), разность двух приближений (dy = y2-y1)
        """
        A, G, g, b, e = self.A, self.G, self.gamma, self.b, self.e
        c = np.sum(A, axis=1)
        k = np.zeros((np.size(b), len(y)))
        I = np.eye(len(y))
        coefk = I - g * dt * ode.jacobian(t, y)
        for i in range(len(A)):
            leftside = dt * ode(t + c[i] * dt, y + np.dot(A[i], k)) + dt * ode.jacobian(t, y).dot(np.dot(G[i], k))
            k[i] = np.linalg.inv(coefk) @ leftside
        # print(np.linalg.inv(coefk) @ leftside)
        # print(b.shape)
        # print(np.array(k).shape)
        # print(e.shape)
        return y + b @ k, e @ k
        # return y1, dy
