from ..core import SkeletonCorrector
from ..utils import Human
import numpy as np


class SwapCorrector(SkeletonCorrector):
    STATE_RESET = 0
    STATE_READY = 1
    STATE_WORKING = 2

    def __init__(self, points_pair: tuple, p_visible=0.1, tolerance=0.3):
        """
        tolerance : float in range [0, 1] - a distance between points is used to calculate tolerance interval.
        dist = abs(x1 - x2)
        if abs(x1 - x2_prev) < dist * tolerance:
            # point has swapped
            x1 = x1_prev
        """
        self.p1_id = points_pair[0]
        self.p2_id = points_pair[1]
        self.p_visible = p_visible
        self.tolerance = tolerance

        self.state = SwapCorrector.STATE_RESET

        self.correction_methods = {
            SwapCorrector.STATE_RESET: self._correction_reset,
            SwapCorrector.STATE_WORKING: self._correction_working,
        }

        # Points statistics
        self.x1_prev = None
        self.x2_prev = None
        self.x1_speed = None
        self.x2_speed = None
        self.side_prev = None

    def __call__(self, skeletons):
        new = []
        for skeleton in skeletons:
            new_skeleton = self.correct(skeleton.to_np())
            new.append(Human.from_array(new_skeleton))
        return new

    def correct(self, skeleton):
        correction_method = self.correction_methods[self.state]
        # Gather points
        p1, p2 = skeleton[self.p1_id].copy(), skeleton[self.p2_id].copy()
        # Do correction
        p1, p2 = correction_method(p1, p2)
        # Assign corrected points
        skeleton[self.p1_id] = p1
        skeleton[self.p2_id] = p2
        return skeleton

    # --- Statistics gathering (speed, position, side)

    def _correction_reset(self, p1, p2):
        if p1[2] < self.p_visible or p2[2] < self.p_visible:
            return p1, p2
        self.x1_prev = p1[0]
        self.x2_prev = p2[0]
        self.side_prev = np.sign(p1[0] - p2[0])

        self.state = SwapCorrector.STATE_WORKING
        return p1, p2

    # --- Actual swap correction

    def _correction_working(self, p1, p2):
        if p1[2] < self.p_visible or p2[2] < self.p_visible:
            # print('Some points are invisible. Reset state')
            self.state = SwapCorrector.STATE_RESET
            return p1, p2

        corrected = False
        # Equals True if the points has swapped
        swap_cur = np.sign(p1[0] - p2[0]) != self.side_prev

        dist = abs(self.x1_prev - self.x2_prev)
        eps = self.tolerance * dist
        if swap_cur and abs(self.x2_prev - p1[0]) < eps:
            corrected = True
            p1[0] = self.x1_prev

        if swap_cur and abs(self.x1_prev - p2[0]) < eps:
            corrected = True
            p2[0] = self.x2_prev

        if corrected:
            return p1, p2

        self.x1_prev = p1[0]
        self.x2_prev = p2[0]

        self.side_prev = np.sign(p1[0] - p2[0])
        return p1, p2