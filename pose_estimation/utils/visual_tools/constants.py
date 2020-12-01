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


CONNECT_INDEXES = [
    # head
    [1, 2],
    [2, 4],
    [1, 3],
    [3, 5],
    # body
    # left
    [1, 7],
    [7, 9],
    [9, 11],
    [11, 22],
    [11, 23],
    # right
    [1, 6],
    [6, 8],
    [8, 10],
    [10, 20],
    [10, 21],
    # center
    [1, 0],
    [0, 12],
    [0, 13],
    # legs
    # left
    [13, 15],
    [15, 17],
    [17, 19],
    # right
    [12, 14],
    [14, 16],
    [16, 18]
]

