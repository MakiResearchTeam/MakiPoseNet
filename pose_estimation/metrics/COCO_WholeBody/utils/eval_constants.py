# Copyright (C) 2020  Igor Kilbas, Danil Gribanov
#
# This file is part of MakiPoseNet.
#
# MakiPoseNet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiPoseNet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
# Constants, aka k in formule

BODY_K = [
    .026, .025, .025, .035, .035, .079, .079, .072, .072,
    .062, .062, 0.107, 0.107, .087, .087, .089, .089
]

FOOT_K = [
    0.068, 0.066, 0.066, 0.092, 0.094, 0.094
]

FACE_K = [
    0.042, 0.043, 0.044, 0.043, 0.040, 0.035, 0.031, 0.025,
    0.020, 0.023, 0.029, 0.032, 0.037, 0.038, 0.043,
    0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012, 0.012,
    0.011, 0.011, 0.013, 0.015, 0.009, 0.007, 0.007,
    0.007, 0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011,
    0.009, 0.011, 0.009, 0.007, 0.013, 0.008, 0.011,
    0.012, 0.010, 0.034, 0.008, 0.008, 0.009, 0.008, 0.008,
    0.007, 0.010, 0.008, 0.009, 0.009, 0.009, 0.007,
    0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01, 0.008
]

LEFT_HAND_K = [
    0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
    0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021,
    0.032, 0.02, 0.019, 0.022, 0.031
]

RIGHT_HAND_K = [
    0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025,
    0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
    0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031
]

ALL_K = BODY_K + FOOT_K + FACE_K + LEFT_HAND_K + RIGHT_HAND_K

# Constants for our skelet with 24 keypoints
MAKI_SKELET_K = [
    # center of the body
    BODY_K[11],
    # neck
    BODY_K[5],
    # face
    *BODY_K[1:5],
    # body
    *BODY_K[5:],
    # foot
    BODY_K[-2],
    BODY_K[-1],
    # hands
    LEFT_HAND_K[4],
    LEFT_HAND_K[-1],
    RIGHT_HAND_K[4],
    RIGHT_HAND_K[-1],
]
