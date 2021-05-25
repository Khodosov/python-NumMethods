import enum
import numpy as np

from utils.ode_collection import ODE
from NumMethods_S3T2.one_step_methods import OneStepMethod, ExplicitEulerMethod


class AdaptType(enum.Enum):
    RUNGE = 0
    EMBEDDED = 1


def fix_step_integration(method: OneStepMethod, ode: ODE, y_start, ts):
    """
    Интегрирование одношаговым методом с фиксированным шагом

    :param method:  одношаговый метод
    :param ode:     СОДУ
    :param y_start: начальное значение
    :param ts:      набор значений t
    :return:        список значений t (совпадает с ts), список значений y
    """
    ys = [y_start]

    for i, t in enumerate(ts[:-1]):
        y = ys[-1]

        y1 = method.step(ode, t, y, ts[i + 1] - t)
        ys.append(y1)

    return ts, ys


def adaptive_step_integration(method: OneStepMethod, ode: ODE, y_start, t_span,
                              adapt_type: AdaptType,
                              atol, rtol):
    """
    Интегрирование одношаговым методом с адаптивным выбором шага.
    Допуски контролируют локальную погрешность:
        err <= atol
        err <= |y| * rtol

    :param method:      одношаговый метод
    :param ode:         СОДУ
    :param y_start:     начальное значение
    :param t_span:      интервал интегрирования (t0, t1)
    :param adapt_type:  правило Рунге (AdaptType.RUNGE) или вложенная схема (AdaptType.EMBEDDED)
    :param atol:        допуск на абсолютную погрешность
    :param rtol:        допуск на относительную погрешность
    :return:            список значений t (совпадает с ts), список значений y
    """
    y = y_start
    t, t_end = t_span

    ys = [y]
    ts = [t]

    p = method.p + 1
    tol = atol + np.linalg.norm(y) * rtol
    rside0 = ode(t, y)
    delta = (1 / max(abs(t), abs(t_end))) ** (p + 1) + np.linalg.norm(rside0) ** (p + 1)
    h1 = (tol / delta) ** (1 / (p + 1))
    u1 = ExplicitEulerMethod().step(ode, t, y, h1)
    tnew = t + h1
    rside0 = ode(tnew, u1)
    delta = (1 / max(abs(t), abs(t_end))) ** (p + 1) + np.linalg.norm(rside0) ** (p + 1)
    h1new = (tol / delta) ** (1 / (p + 1))
    h = min(h1, h1new)
    while t < t_end:
        if t + h > t_end:
            h = t_end - t

        if adapt_type == AdaptType.RUNGE:
            y1 = method.step(ode, t, y, h)
            yhalf = method.step(ode, t, y, h / 2)
            y2 = method.step(ode, t + h / 2, yhalf, h / 2)
            error = (y2 - y1) / (2 ** p - 1)
            ybetter = y2 + error
        else:
            ybetter, error = method.embedded_step(ode, t, y, h)

        if np.linalg.norm(error) < tol:
            ys.append(ybetter)
            ts.append(t + h)
            y = ybetter
            t += h
            # print(t)

        h *= (tol / np.linalg.norm(error)) ** (1 / (p + 1)) * 0.8
    return ts, ys
