from makiflow.core import MakiRestorable
from makiflow.core import MakiLayer
import tensorflow as tf
import numpy as np


class PAFLayerV2(MakiLayer):
    IM_SIZE = 'im_size'
    SIGMA = 'sigma'
    SKELETON = 'skeleton'
    VECTORIZE = 'vectorize'

    RESIZE_TO = 'resize_to'
    PAF_RESIZE = 'paf_resize'

    @staticmethod
    def build(params: dict):
        return PAFLayerV2(
            im_size=params[PAFLayerV2.IM_SIZE],
            sigma=params[PAFLayerV2.SIGMA],
            skeleton=params[PAFLayerV2.SKELETON],
            vectorize=params[PAFLayerV2.VECTORIZE],
            resize_to=params[PAFLayerV2.RESIZE_TO]
        )

    # 90 degrees rotation matrix. Used for generation of orthogonal vector.
    ROT90_MAT = tf.convert_to_tensor(
        np.array([
            [0, 1],
            [-1, 0]
        ]),
        dtype=tf.float32
    )

    def __init__(self, im_size: list, sigma, skeleton, vectorize=False, resize_to=None, name='PAFLayer'):
        """
        Generates part affinity fields for the given `skeleton`.

        Parameters
        ----------
        im_size : 2d tuple
            Contains width and height (h, w) of the image for which to generate the map.
        sigma : float
            Width of the affinity field. Corresponds to the width of the limb.
        skeleton : np.ndarray of shape [n_pairs, 2]
            A numpy array containing indices for pairs of points. Vectors in the PAF
            will be equal the following vector:
            (point1 - point0) / norm(point1 - point0).
        vectorize : bool
            Set to True if you want to vectorize the computation along the batch dimension. May cause
            the OOM error due to high memory consumption.
        resize_to : tuple
            Tuple of (H, W) the size to which the heatmap will be reduced or scaled,
            Using area interpolation

        """
        assert resize_to is None or len(resize_to) == 2

        super().__init__(name, params=[], regularize_params=[], named_params_dict={})
        self.sigma = sigma
        self.resize_to = resize_to
        self.skeleton = skeleton
        self.im_size = [im_size[1], im_size[0]]
        self.vectorize = vectorize
        x_grid, y_grid = np.meshgrid(np.arange(im_size[1]), np.arange(im_size[0]))
        xy_grid = np.stack([x_grid, y_grid], axis=-1)
        self.xy_grid = tf.convert_to_tensor(xy_grid, dtype=tf.float32)

    def forward(self, x, computation_mode=MakiRestorable.TRAINING_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                keypoints, masks = x

                pafs = self.__build_paf_batch(keypoints, masks)

                if self.resize_to is not None:
                    # [batch, h, w, pairs, 2]
                    pafs_shape = pafs.get_shape().as_list()

                    # [batch, h, w, pairs, 2] --> [batch, h, w, pairs * 2]
                    pafs = tf.reshape(pafs, pafs_shape[:3] + [-1])

                    pafs = tf.image.resize_area(
                        pafs,
                        self.resize_to,
                        align_corners=False,
                        name=self.PAF_RESIZE
                    )
                    # [batch, h, w, pairs * 2] --> [batch, h, w, pairs, 2]
                    pafs_shape[1] = self.resize_to[0]
                    pafs_shape[2] = self.resize_to[1]
                    pafs = tf.reshape(pafs, pafs_shape)

        return pafs

    def training_forward(self, x):
        return self.forward(x)

    def to_dict(self):
        return {}

    def __build_paf_batch(self, kp, masks):
        """
        Generates PAF for the given keypoints.

        Parameters
        ----------
        kp : tf.Tensor of shape [batch, c, n_people, 2]
            Tensor of keypoints coordinates.
        masks : tf.Tensor of shape [batch, c, n_people, 1]
        Returns
        -------
        tf.Tensor of shape [batch, h, w, n_pars, 2]
            Tensor of PAFs.
        """

        # ------------------THE SHITTEST CODE EVER----------------------------------------------

        # Gather points along the axis of classes of points.
        # [b, n_pairs, 2, p, 2]
        kp_p = tf.gather(kp, indices=self.skeleton, axis=1)
        # This is needed for proper matrix multiplication during paf generation.
        kp_p = tf.transpose(kp_p, perm=[0, 1, 3, 2, 4])
        kp_p = tf.expand_dims(kp_p, axis=-1)
        # [b, n_pairs, p, 2, 2, 1]
        assert len(kp_p.get_shape()) == 6, f'Expected keypoint pairs dimensionality to be 6, but ' \
            f'got {len(kp_p.get_shape())}. Keypoints shape: {kp_p.get_shape()}'

        # Select masks for corresponding points.
        # [b, n_pairs, 2, p, 1]
        masks_p = tf.gather(masks, indices=self.skeleton, axis=1)
        masks_p = tf.transpose(masks_p, perm=[0, 1, 3, 2, 4])

        def normalize_paf(paf):
            """
            Averages values in the regions where PAF overlaps.
            Parameters
            ----------
            paf : tf.Tensor of shape [n_people, h, w, 2]

            Returns
            -------
            tf.Tensor of shape [h, w, 2]
                Normalized paf tensor.
            """
            # [n_people, h, w]
            magnitudes = tf.reduce_sum(paf * paf, axis=-1)
            ones = tf.ones_like(magnitudes, dtype='float32')
            zeros = tf.zeros_like(magnitudes, dtype='float32')
            non_zero_regions = tf.where(magnitudes > 1e-3, ones, zeros)
            # [h, w]
            normalization_mask = tf.reduce_sum(non_zero_regions, axis=0)
            # Set zeros to ones to avoid division by zero.
            # Don't change other regions
            ones = tf.ones_like(normalization_mask, dtype='float32')
            normalization_mask = tf.where(normalization_mask > 1e-3, normalization_mask, ones)
            # [h, w]
            paf = tf.reduce_sum(paf, axis=0)
            print(paf)
            result = paf / tf.expand_dims(normalization_mask, axis=-1)
            print(result)
            return result

        # [h, w, 2]
        fn_p = lambda kp, masks: normalize_paf(
            PAFLayerV2.__build_paf_mp(
                kp, masks,  # [p, 2, 2, 1]
                destination_call=self.__build_paf
            )
        )

        def shape_fixer(t):
            h, w, _ = self.xy_grid.get_shape()
            t.set_shape(shape=[len(self.skeleton), h, w, 2])
            return t

        # [n_pairs, h, w, 2]
        fn_np = lambda kp, masks: shape_fixer(PAFLayerV2.__build_paf_mp(
            kp, masks,  # [n_pairs, p, 2, 2, 1]
            destination_call=fn_p
        ))

        # [b, n_pairs, h, w, 2]
        fn_b = lambda kp, masks: PAFLayerV2.__build_paf_mp(
            kp, masks,
            destination_call=fn_np
        )
        # Decide whether to perform calculation in a batch dimension.
        # May be faster, but requires more memory.
        if self.vectorize:  # [b, c, p, 2]
            print('Using vectorized_map.')
            pafs = fn_b(kp_p, masks_p)
            pafs = tf.transpose(pafs, perm=[0, 2, 3, 1, 4])
            return pafs
        else:
            # Requires less memory, but runs slower
            print('Using map_fn.')
            # The map_fn function passes in a list of unpacked tensors
            # along the first dimension. Therefore, we need to take those tensors
            # out of the list.
            fn = lambda kp_masks: [fn_np(kp_masks[0], kp_masks[1]), 0]
            pafs, _, = tf.map_fn(
                fn,
                [kp_p, masks_p]  # [b, n_pairs, p, 2, 2, 1]
            )
            print('after map_fn:', pafs)
            pafs = tf.transpose(pafs, perm=[0, 2, 3, 1, 4])
            return pafs

    @staticmethod
    def __build_paf_mp(kp, masks, destination_call):
        """
        The hetmaps generation is factorized down to single keypoint heatmap generation.
        Nested calls of this method allow for highly optimized vectorized computation of multiple maps.

        Parameters
        ----------
        kp : tf.Tensor of shape [..., 2]
            A keypoint (x, y) for which to build the heatmap.
        xy_grid : tf.Tensor of shape [h, w, 2]
            A coordinate grid for the image tensor.
        delta : tf.float32
            Radius of the classification (heat) region.
        destination_call : method pointer
            Used for nested calling to increase the dimensionality of the computation.
        """
        # Vectorized map unpackes tensors from the input sequence (list in this case) along their
        # first dimension, but passes a list of unpacked tensors. Therefore, we need to take them
        # out from the list.
        fn = lambda _kp_masks: destination_call(_kp_masks[0], _kp_masks[1])
        maps = tf.vectorized_map(fn, [kp, masks])
        return maps

    def __build_paf(self, p1p2, points_mask):
        """
        Build PAF for the given 2 points `p1p2` on the `xy_grid`.

        Parameters
        ----------
        p1p2 : tf.Tensor of shape [2, 2, 1]
            Points between which to generate the field. Vectors will points from
            p1 to p2.
        point_mask : tf.Tensor of shape [2, 1]
            A mask determining whether the keypoints are labeled. 0 - no label, 1 - there is a label.

        Returns
        -------
        tf.Tensor of shape [h, w, 2]
            The generated PAF.
        """
        xy_grid = self.xy_grid
        h, w, _ = xy_grid.get_shape().as_list()
        p1 = p1p2[0]
        p2 = p1p2[1]
        l = tf.maximum(tf.linalg.norm(p2 - p1), 1e-5)
        # v - [2, 1]
        v = (p2 - p1) / l

        xy_flat = tf.reshape(xy_grid, [-1, 2, 1])

        # Define the border and determine the points within it
        min_x = tf.reduce_max([0.0, tf.reduce_min(p1p2[:, 0]) - self.sigma])
        max_x = tf.reduce_min([w, tf.reduce_max(p1p2[:, 0]) + self.sigma])
        min_y = tf.reduce_max([0.0, tf.reduce_min(p1p2[:, 1]) - self.sigma])
        max_y = tf.reduce_min([h, tf.reduce_max(p1p2[:, 1]) + self.sigma])
        inside_border_x = tf.logical_and(xy_flat[:, 0] > min_x, xy_flat[:, 0] < max_x)
        inside_border_y = tf.logical_and(xy_flat[:, 1] > min_y, xy_flat[:, 1] < max_y)
        inside_border = tf.logical_and(inside_border_x, inside_border_y)
        inside_border = tf.reshape(inside_border, shape=[h, w])
        inside_border = tf.stack([inside_border, inside_border], axis=-1)

        # Determine the points lying on the line between p1 and p2
        bec_x = xy_flat[:, 0] - p1[0]
        bec_y = xy_flat[:, 1] - p1[1]
        dist = tf.abs(bec_x * v[1] - bec_y * v[0])
        dist = tf.reshape(dist, shape=[h, w])
        dist = tf.stack([dist, dist], axis=-1)

        cond = tf.logical_and(dist < self.sigma, inside_border)
        cond = tf.cast(cond, dtype='float32')
        vs = tf.ones_like(xy_grid) * v[:, 0]
        paf = vs * cond

        return paf * tf.reduce_min(points_mask)


if __name__ == '__main__':
    from makiflow.layers import InputLayer

    CONNECT_INDEXES_FOR_PAFF = [
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
        [16, 18],
        # Additional limbs
        [5, 7],
        [4, 6],
    ]
    """
    kp : tf.Tensor of shape [batch, c, n_people, 2]
            Tensor of keypoints coordinates.
    masks : tf.Tensor of shape [batch, c, n_people, 1]
    """
    im_size = [512, 512]
    paf_sigma = 20
    keypoints = InputLayer(input_shape=[32, 24, 8, 2], name='keypoints')
    masks = InputLayer(input_shape=[32, 24, 8, 1], name='keypoints')

    paf_layer = PAFLayerV2(
        im_size=im_size,
        sigma=paf_sigma,
        skeleton=CONNECT_INDEXES_FOR_PAFF
    )
    paf = paf_layer([keypoints, masks])

    sess = tf.Session()
    paf_shape = sess.run(
        tf.shape(paf.get_data_tensor()),
        feed_dict={
            keypoints.get_data_tensor(): np.random.randn(32, 24, 8, 2),
            masks.get_data_tensor(): np.random.randn(32, 24, 8, 1)
        }
    )
    print(paf)
    print(paf_shape)
    import matplotlib

    # For some reason matplotlib doesn't want to show the plot when it is called from PyCharm
    matplotlib.use('TkAgg')
    import seaborn as sns
    import matplotlib.pyplot as plt
    import math

    def put_paf_on_plane(vectormap, countmap, plane_idx, x1, y1, x2, y2,
                         threshold, height, width):

        min_x = max(0, int(round(min(x1, x2) - threshold)))
        max_x = min(width, int(round(max(x1, x2) + threshold)))

        min_y = max(0, int(round(min(y1, y2) - threshold)))
        max_y = min(height, int(round(max(y1, y2) + threshold)))

        vec_x = x2 - x1
        vec_y = y2 - y1

        norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
        if norm < 1e-8:
            return

        vec_x /= norm
        vec_y /= norm

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                bec_x = x - x1
                bec_y = y - y1
                dist = abs(bec_x * vec_y - bec_y * vec_x)

                if dist > threshold:
                    continue

                cnt = countmap[y][x][plane_idx]

                if cnt == 0:
                    vectormap[y][x][plane_idx * 2 + 0] = vec_x
                    vectormap[y][x][plane_idx * 2 + 1] = vec_y
                else:
                    vectormap[y][x][plane_idx * 2 + 0] = (vectormap[y][x][plane_idx * 2 + 0] * cnt + vec_x) / (cnt + 1)
                    vectormap[y][x][plane_idx * 2 + 1] = (vectormap[y][x][plane_idx * 2 + 1] * cnt + vec_y) / (cnt + 1)

                countmap[y][x][plane_idx] += 1
        return vectormap

    # PLOT THE KERAS IMPLEMENTATION
    x = np.linspace(0, 512, 512, dtype='float32')
    x, y = np.meshgrid(x, x)
    grid = np.stack([x, y], axis=-1)
    countmap = np.zeros_like(grid)
    paf = put_paf_on_plane(countmap, countmap, 0, 256, 256, 384, 384, paf_sigma, 512, 512)

    sns.heatmap(paf[..., 0] ** 2 + paf[..., 1] ** 2)
    plt.show()

    # PLOT OUR IMPLEMENTATION V2
    p1p2 = np.array([
        [256, 256], [384, 384]
    ], dtype='float32').reshape(2, 2, 1)
    tf_paf = paf_layer._PAFLayerV2__build_paf(p1p2, np.ones(shape=[2, 1], dtype='float32'))
    tf_paf = sess.run(tf_paf)
    sns.heatmap(tf_paf[..., 0] ** 2 + tf_paf[..., 1] ** 2)
    plt.show()

    from pose_estimation.model.training_modules.training_layers import PAFLayer
    paf_layer = PAFLayer(
        im_size=im_size,
        sigma=paf_sigma,
        skeleton=CONNECT_INDEXES_FOR_PAFF
    )


