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



def rescale_image(
        image_size: list, resize_to: list,
        min_image_size: list, use_force_resize=False) -> list:
    """
    Rescale image to minimum size,
    if one of the dimension of the image (height and width) is smaller than `min_image_size`

    """
    h, w = image_size

    if not use_force_resize:
        # Force resize use only min_image_size params
        assert not (min_image_size[0] is None and min_image_size[1] is None)
        if h is None:
            h_is_smaller = False
        elif w is None:
            h_is_smaller = True
        else:
            h_is_smaller = h < w

        if h_is_smaller:
            scale = [
                1.0,
                min_image_size[0] / h
            ]
        else:
            scale = [
                min_image_size[1] / w,
                1.0
            ]
    else:
        # Use resize_to and takes into account min_image_size
        x_scale = 1.0
        y_scale = 1.0

        if min_image_size[0] is None:
            min_image_size[0] = resize_to[0]

        if min_image_size[1] is None:
            min_image_size[1] = resize_to[1]

        if h < min_image_size[0]:
            if resize_to[0] is not None and resize_to[0] > min_image_size[0]:
                y_scale = resize_to[0] / h
            else:
                y_scale = min_image_size[0] / h

        if w < min_image_size[1]:
            if resize_to[1] is not None and resize_to[1] > min_image_size[1]:
                x_scale = resize_to[1] / w
            else:
                x_scale = min_image_size[1] / w

        scale = [
            x_scale,
            y_scale
        ]

    return scale
