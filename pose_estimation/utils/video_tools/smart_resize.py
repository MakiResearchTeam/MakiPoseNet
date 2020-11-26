

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
    Biggest dimension in `image_size` will be scaled to certain dimension in `min_image_size`,
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

