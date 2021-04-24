import math
import numpy as np

from ..core import SkeletonCorrector
from ..utils import Human


class LowPassFilter(object):

    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.reset_values()

    def __setAlpha(self, alpha):
        alpha = float(alpha)
        if alpha <= 0 or alpha > 1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]" % alpha)
        self.__alpha = alpha

    def reset_values(self):
        self.__y = self.__s = None

    def __call__(self, value, timestamp=None, alpha=None):
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha * value + (1.0 - self.__alpha) * self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self):
        return self.__y


class OneEuroFilter(object):
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        if freq <= 0:
            raise ValueError("freq should be >0")
        if mincutoff <= 0:
            raise ValueError("mincutoff should be >0")
        if dcutoff <= 0:
            raise ValueError("dcutoff should be >0")
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None

    def __alpha(self, cutoff):
        te = 1.0 / self.__freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        # ---- update the sampling frequency based on timestamps
        if self.__lasttime and timestamp:
            self.__freq = 1.0 / (timestamp - self.__lasttime)
        self.__lasttime = timestamp
        # ---- estimate the current variation per second
        prev_x = self.__x.lastValue()
        dx = 0.0 if prev_x is None else (x - prev_x) * self.__freq  # FIXME: 0.0 or value?
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        # ---- use it to update the cutoff frequency
        cutoff = self.__mincutoff + self.__beta * math.fabs(edx)
        # ---- filter the given value
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))

    def reset_values(self):
        self.__x.reset_values()
        self.__dx.reset_values()


class OneEuro(SkeletonCorrector):
    """
    This correction module is based in 1 euro algorithm
    For mode details refer to: https://hal.inria.fr/hal-00670496/document
    """

    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0, num_kp=24):
        self._euro_list = [OneEuroFilter(freq, mincutoff, beta, dcutoff) for _ in range(num_kp * 2)]

    def __call__(self, skeletons: list) -> list:
        # (N, 3)
        single_human = skeletons[0].to_np(0.3)
        points_xy = np.zeros(single_human[:, :-1].shape).astype(np.float32)
        for i in range(len(single_human)):
            if single_human[i][-1] < 1e-3:
                self._euro_list[i * 2].reset_values()
                self._euro_list[i * 2 + 1].reset_values()
            else:
                points_xy[i, 0] = self._euro_list[i * 2](single_human[i][0])
                points_xy[i, 1] = self._euro_list[i * 2 + 1](single_human[i][1])

        return [Human.from_array(np.concatenate([points_xy, single_human[:, 2:3]], axis=-1))]
