from ..core import SkeletonCorrector
from ..utils import Human
import numpy as np


class SwapCorrectorV2(SkeletonCorrector):
    STATE_RESET = 0
    STATE_READY = 1
    STATE_WORKING = 2

    def __init__(self, points_pair, momentum=0.9, acceleration=2, p_visible=0.1, etalon=None):
        """
        Performs prediction-correction of the five `points_pair`.
        First, the corrector checks is the human is turned sidewards via looking at the distance
        between `etalon` points.
        if abs(p1 - p2) < abs(et1 - et2) * et3:
            # human turned sidewards, perform reset
        else:
            # perform correction
        Second, it gathers statistics about the motion of the points.
        Third, it predicts current position of the points via gathered statistics from the previous step. If the swap
        wasn't predicted but actually happened, the new points are claimed to wrong and are being replaced with the ones
        from the previous step.

        Parameters
        ----------
        points_pair : array of len 2
            A pair of indices of the points that are prone to erroneous swap.
        momentum : float
            The speed of the points is computed as speed(t) = speed(t-1)*momentum + measured_speed*(1 - momentum).
        acceleration : float
            Current position of a point is being predicted as cur_x = prev_x + acceleration * speed
        p_visible : float
            If any of the points in `points_pair` have a probability less than `p_visible`, the corrector performs
            reset as one of the points was considered to be absent.
        etalon : array of len 3
            Contains a pair of indices for etalon points and a tolerance factor (float in [0, 1]). Etalon points
            are such points that are constantly distant (the same distance) from each other. For example,
            left knee and left foot. Distance between those points is used to determine whether the person is turned
            sidewards (or to determine the proximity in which points from `points_pair` are lying; if the points are
            close enough, the corrector performs reset).
        """
        self.p1_id = points_pair[0]
        self.p2_id = points_pair[1]
        self.mu = momentum
        self.acceleration = acceleration
        self.p_visible = p_visible
        self.etalon = etalon
        self.state = SwapCorrectorV2.STATE_RESET

        self.correction_methods = {
            SwapCorrectorV2.STATE_RESET: self._correction_reset,
            SwapCorrectorV2.STATE_READY: self._correction_ready,
            SwapCorrectorV2.STATE_WORKING: self._correction_working,
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
        if self.etalon:
            e1, e2 = skeleton[self.etalon[0]].copy(), skeleton[self.etalon[1]].copy()
            if abs(p1[0] - p2[0]) < abs(e1[0] - e2[0]) * self.etalon[2]:
                correction_method = lambda x1, x2: (x1, x2)
                self.state = SwapCorrectorV2.STATE_RESET
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

        self.state = SwapCorrectorV2.STATE_READY
        return p1, p2

    def _correction_ready(self, p1, p2):
        if p1[2] < self.p_visible or p2[2] < self.p_visible:
            self.state = SwapCorrectorV2.STATE_RESET
            return p1, p2

        self.x1_speed = p1[0] - self.x1_prev
        self.x2_speed = p2[0] - self.x2_prev
        self.x1_prev = p1[0]
        self.x2_prev = p2[0]

        self.side_prev = np.sign(p1[0] - p2[0])

        self.state = SwapCorrectorV2.STATE_WORKING

        return p1, p2

    # --- Actual swap correction

    def _correction_working(self, p1, p2):
        if p1[2] < self.p_visible or p2[2] < self.p_visible:
            # print('Some points are invisible. Reset state')
            self.state = SwapCorrectorV2.STATE_RESET
            return p1, p2

        # Perform prediction
        x1_pred = self.x1_prev + self.x1_speed * self.acceleration
        x2_pred = self.x2_prev + self.x2_speed * self.acceleration

        # If the side has changed, than we accept any condition
        side_pred = np.sign(x1_pred - x2_pred)
        # Equals True if a swap has been predicted
        swap_pred = side_pred != self.side_prev
        # Equals True if the points has swapped
        swap_cur = np.sign(p1[0] - p2[0]) != self.side_prev

        if swap_cur and not swap_pred:
            p1, p2 = p2, p1

        self.x1_speed = self.mu * self.x1_speed + (1 - self.mu) * (p1[0] - self.x1_prev)
        self.x2_speed = self.mu * self.x2_speed + (1 - self.mu) * (p2[0] - self.x2_prev)
        self.x1_prev = p1[0]
        self.x2_prev = p2[0]

        self.side_prev = np.sign(p1[0] - p2[0])

        return p1, p2

