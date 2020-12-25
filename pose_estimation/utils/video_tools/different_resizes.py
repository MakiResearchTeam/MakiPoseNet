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


def rescale_image_keep_relation(
        image_size: tuple,
        limit_size: tuple,
        use_max=True) -> list:
    """
    Calculate final image size according to `min_image_size`
    Biggest dimension (if `use_max` = True, otherwise lowest dimension will has same effect)
    in `image_size` will be scaled to certain dimension in `min_image_size`,
    Relation between origin Height and Width will be saved

    Parameters
    ----------
    image_size : tuple
        (H, W), tuple of Height and Width of image which will be scaled
    limit_size : tuple
        (H_min, W_min), tuple of Height and Width of minimum image size to certain dimension
    use_max : bool
        If true, when maximum dimension from `image_size` will has value from `limit_size`,
        otherwise, minimum dimension will has value from `limit_size`,
        other dimension will be scaled according to relation of H and W of `image_size`

    Returns
    -------
    list
        (H_final, W_final)
    
    """
    h, w = image_size
    relation = h / w

    h_limit, w_limit = limit_size

    if use_max:
        h_is_smaller = h < w

        if h_is_smaller:
            hw = [
                int(relation * w_limit),  # h
                int(w_limit)              # w
            ]
        else:
            hw = [
                int(h_limit),             # h
                int(h_limit / relation)   # w
            ]
    else:
        h_is_smaller = h < w

        if h_is_smaller:
            hw = [
                int(h_limit),             # h
                int(h_limit / relation)   # w
            ]
        else:
            hw = [
                int(relation * w_limit),  # h
                int(w_limit)              # w
            ]

    return hw


def scales_image_single_dim_keep_dims(
        image_size: tuple,
        resize_to: int,
        resize_h=True) -> list:
    """
    Resize single dimension (by default h) and keep resolution for other

    Parameters
    ----------
    image_size : tuple
        (H, W)
    resize_to : int
        Final value for chosen dims
    resize_h : bool
        If equal to True, then will be calculated scales for H dimension (i.e. index zero in `images_size`)

    Returns
    -------
    xy_scales : list
        Scales for x (Width) and y (Height) dimensions

    """
    h, w = image_size

    if resize_h:
        relation = h / w

        xy_scale = [
            (resize_to / w) / relation,
            resize_to / h
        ]
    else:
        relation = image_size[0] / image_size[1]

        xy_scale = [
            (resize_to / w),
            (resize_to / h) * relation
        ]

    return xy_scale


if __name__ == "__main__":

    image_size = (720, 1280)

    xy_scale = scales_image_single_dim_keep_dims(image_size, resize_to=298)

    print(f'h: {int(image_size[0]*xy_scale[1])} w: {int(image_size[1]*xy_scale[0])}')
