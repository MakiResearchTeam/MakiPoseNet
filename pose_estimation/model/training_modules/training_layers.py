from makiflow.core import MakiRestorable
from makiflow.core import MakiLayer
import tensorflow as tf
import numpy as np


class BinaryHeatmapLayer(MakiLayer):

    IM_SIZE = 'im_size'
    DELTA = 'delta'
    MAP_DTYPE = 'map_dtype'
    VECTORIZE = 'vectorize'
    SCALE_KEYPOINTS = 'scale_keypoints'

    RESIZE_TO = 'resize_to'
    HEATMAP_RESIZE = 'heatmap_resize'

    @staticmethod
    def build(params: dict):
        return BinaryHeatmapLayer(
            im_size=params[BinaryHeatmapLayer.IM_SIZE],
            delta=params[BinaryHeatmapLayer.DELTA],
            map_dtype=params[BinaryHeatmapLayer.MAP_DTYPE],
            vectorize=params[BinaryHeatmapLayer.VECTORIZE],
            resize_to=params[PAFLayer.RESIZE_TO]
        )

    def to_dict(self):
        raise NotImplementedError()
    
    def __init__(
            self,
            im_size: list,
            delta,
            map_dtype=tf.float32,
            vectorize=False,
            resize_to=None,
            name='BinaryHeatmapLayer'):
        """
        Generates hard keypoint maps using highly optimized vectorization.

        Parameters
        ----------
        im_size : 2d tuple
            Contains width and height (h, w) of the image for which to generate the map.
        delta : float
            Radius of a label-circle around the keypoint.
        map_dtype : tf.dtype
            Dtype of the generated map. Use tf.int32 for binary classification and tf.float32 for
            regression.
        vectorize : bool
            Set to True if you want to vectorize the computation along the batch dimension. May cause
            the OOM error due to high memory consumption.
        resize_to : tuple
            Tuple of (H, W) the size to which the heatmap will be reduced or scaled,
            Using area interpolation
        """
        assert resize_to is None or len(resize_to) == 2

        super().__init__(name, params=[], regularize_params=[], named_params_dict={})
        self.im_size = [im_size[1], im_size[0]]
        self.resize_to = resize_to
        self.delta = tf.convert_to_tensor(delta, dtype=tf.float32)
        self.map_dtype = map_dtype
        self.vectorize = vectorize
        # Prepare the grid.
        x_grid, y_grid = np.meshgrid(np.arange(im_size[1]), np.arange(im_size[0]))
        xy_grid = np.stack([x_grid, y_grid], axis=-1)
        self.xy_grid = tf.convert_to_tensor(xy_grid, dtype=tf.float32)

    def forward(self, x, computation_mode=MakiRestorable.TRAINING_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                keypoints, masks = x

                maps = self.__build_heatmap_batch(keypoints, masks, self.delta)
                maps = self.__add_background_heatmap(maps)

                if self.resize_to is not None:
                    maps = tf.image.resize_area(
                        maps,
                        self.resize_to,
                        align_corners=False,
                        name=self.HEATMAP_RESIZE
                    )
        return maps

    def training_forward(self, x):
        return self.forward(x)

    def __build_heatmap_batch(self, kp, masks, delta):
        # Build maps for keypoints of the same class for multiple people
        # and then aggregate generated maps.
        # [h, w]
        fn_p = lambda kp, masks, delta: tf.reduce_max(
            BinaryHeatmapLayer.__build_heatmap_mp(
                kp,
                masks, 
                delta,
                destination_call=self.__build_heatmap
            ),
            axis=0
        )
        # Build maps for keypoints of multiple classes.
        # [c, h, w]
        fn_c = lambda kp, masks, delta: BinaryHeatmapLayer.__build_heatmap_mp(
            kp,
            masks, 
            delta,
            destination_call=fn_p
        )
        # Build a batch of maps.
        # [b, c, h, w]
        fn_b = lambda kp, masks, delta: BinaryHeatmapLayer.__build_heatmap_mp(
            kp,
            masks, 
            delta,
            destination_call=fn_c
        )
        
        # Decide whether to perform calucalation in a batch dimension.
        # May be faster, but requires more memory.
        if len(kp.get_shape()) == 4 and self.vectorize:            # [b, c, p, 2]
            print('Using vectorized_map.')
            heatmaps = fn_b(kp, masks, delta)
            heatmaps = tf.transpose(heatmaps, perm=[0, 2, 3, 1])
            return heatmaps
        elif len(kp.get_shape()) == 4 and not self.vectorize: 
            # Requires less memory, but runs slower
            print('Using map_fn.')
            fn = lambda kp_masks: [fn_c(kp_masks[0], kp_masks[1], delta), 0]
            heatmaps, _ = tf.map_fn(
                fn,
                [kp, masks]
            )
            heatmaps = tf.transpose(heatmaps, perm=[0, 2, 3, 1])
            return heatmaps
        else:
            message = f'Expected keypoints dimensionality to be 4, but got {len(kp.get_shape())}.' + \
                f'Keypoints shape: {kp.get_shape()}'
            raise Exception(message)

    @staticmethod
    def __build_heatmap_mp(kp, masks, delta, destination_call):
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
        fn = lambda _kp_masks: destination_call(_kp_masks[0], _kp_masks[1], delta)
        maps = tf.vectorized_map(
            fn, 
            [kp, masks]
        )
        return maps

    def __build_heatmap(self, kp, masks, delta):
        """
        Builds a hard classification heatmap for a single keypoint `kp`.
        Parameters
        ----------
        kp : tf.Tensor of shape [2]
            A keypoint (x, y) for which to build the heatmap.
        masks : tf.Tensor of shape [2, 1]
        xy_grid : tf.Tensor of shape [h, w, 2]
            A coordinate grid for the image tensor.
        delta : tf.float32
            Radius of the classification (heat) region.
        """
        xy_grid = self.xy_grid
        grid_size = xy_grid.get_shape()[:2]
        heatmap = tf.ones((grid_size[0], grid_size[1]), dtype=self.map_dtype)

        bool_location_map = (xy_grid[..., 0] - kp[0])**2 + (xy_grid[..., 1] - kp[1])**2 < delta**2
        bool_location_map = tf.cast(bool_location_map, dtype=self.map_dtype)
        masks = tf.cast(masks, dtype=self.map_dtype)
        return heatmap * bool_location_map * tf.reduce_min(masks)

    # noinspection PyMethodMayBeStatic
    def __add_background_heatmap(self, heatmap):
        # mask - [b, h, w, c]
        # [b, h, w]
        heatmap_reduced = tf.reduce_max(heatmap, axis=-1)
        # [b, h, w, 1]
        heatmap_reduced = tf.expand_dims(heatmap_reduced, axis=-1)
        background_heatmap = 1.0 - heatmap_reduced
        return tf.concat([heatmap, background_heatmap], axis=-1)


class GaussHeatmapLayer(MakiLayer):
    IM_SIZE = 'im_size'
    DELTA = 'delta'
    VECTORIZE = 'vectorize'
    SCALE_KEYPOINTS = 'scale_keypoints'

    RESIZE_TO = 'resize_to'
    HEATMAP_RESIZE = 'heatmap_resize'

    @staticmethod
    def build(params: dict):
        return GaussHeatmapLayer(
            im_size=params[GaussHeatmapLayer.IM_SIZE],
            delta=params[GaussHeatmapLayer.DELTA],
            vectorize=params[GaussHeatmapLayer.VECTORIZE],
            resize_to=params[PAFLayer.RESIZE_TO]
        )

    def __init__(self, im_size: list, delta, vectorize=False, resize_to=None, name='GaussHeatmapLayer'):
        """
        Generates hard keypoint maps using highly optimized vectorization.

        Parameters
        ----------
        im_size : 2d tuple
            Contains width and height (h, w) of the image for which to generate the map.
        delta : float
            Defines the spreadout of the heat around the point.
        vectorize : bool
            Set to True if you want to vectorize the computation along the batch dimension. May cause
            the OOM error due to high memory consumption.
        resize_to : tuple
            Tuple of (H, W) the size to which the heatmap will be reduced or scaled,
            Using area interpolation

        """
        assert resize_to is None or len(resize_to) == 2

        super().__init__(name, params=[], regularize_params=[], named_params_dict={})
        self.im_size = [im_size[1], im_size[0]]
        self.resize_to = resize_to
        self.delta = tf.convert_to_tensor(delta, dtype=tf.float32)
        self.vectorize = vectorize
        # Prepare the grid.
        x_grid, y_grid = np.meshgrid(np.arange(im_size[1]), np.arange(im_size[0]))
        xy_grid = np.stack([x_grid, y_grid], axis=-1)
        self.xy_grid = tf.convert_to_tensor(xy_grid, dtype=tf.float32)

    def forward(self, x, computation_mode=MakiRestorable.TRAINING_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                keypoints, masks = x

                maps = self.__build_heatmap_batch(keypoints, masks, self.delta)
                maps = self.__add_background_heatmap(maps)

                if self.resize_to is not None:
                    maps = tf.image.resize_area(
                        maps,
                        self.resize_to,
                        align_corners=False,
                        name=self.HEATMAP_RESIZE
                    )
        return maps

    def training_forward(self, x):
        return self.forward(x)

    def to_dict(self):
        raise NotImplementedError()

    def __build_heatmap_batch(self, kp, masks, delta):
        # Build maps for keypoints of the same class for multiple people
        # and then aggregate generated maps.
        # [h, w]
        fn_p = lambda kp, masks, delta: tf.reduce_max(
            GaussHeatmapLayer.__build_heatmap_mp(
                kp,
                masks, 
                delta,
                destination_call=self.__build_heatmap
            ),
            axis=0
        )
        # Build maps for keypoints of multiple classes.
        # [c, h, w]
        fn_c = lambda kp, masks, delta: GaussHeatmapLayer.__build_heatmap_mp(
            kp,
            masks, 
            delta,
            destination_call=fn_p
        )
        # Build a batch of maps.
        # [b, c, h, w]
        fn_b = lambda kp, masks, delta: GaussHeatmapLayer.__build_heatmap_mp(
            kp,
            masks, 
            delta,
            destination_call=fn_c
        )
        
        # Decide whether to perform calucalation in a batch dimension.
        # May be faster, but requires more memory.
        if len(kp.get_shape()) == 4 and self.vectorize:            # [b, c, p, 2]
            print('Using vectorized_map.')
            heatmaps = fn_b(kp, masks, delta)
            heatmaps = tf.transpose(heatmaps, perm=[0, 2, 3, 1])
            return heatmaps
        elif len(kp.get_shape()) == 4 and not self.vectorize: 
            # Requires less memory, but runs slower
            print('Using map_fn.')
            fn = lambda kp_masks: [fn_c(kp_masks[0], kp_masks[1], delta), 0]
            heatmaps, _ = tf.map_fn(
                fn,
                [kp, masks]
            )
            heatmaps = tf.transpose(heatmaps, perm=[0, 2, 3, 1])
            return heatmaps
        else:
            message = f'Expected keypoints dimensionality to be 4, but got {len(kp.get_shape())}.' + \
                f'Keypoints shape: {kp.get_shape()}'
            raise Exception(message)

    # noinspection PyMethodMayBeStatic
    def __add_background_heatmap(self, heatmap):
        # mask - [b, h, w, c]
        # [b, h, w]
        heatmap_reduced = tf.reduce_max(heatmap, axis=-1)
        # [b, h, w, 1]
        heatmap_reduced = tf.expand_dims(heatmap_reduced, axis=-1)
        background_heatmap = 1.0 - heatmap_reduced
        return tf.concat([heatmap, background_heatmap], axis=-1)

    @staticmethod
    def __build_heatmap_mp(kp, masks, delta, destination_call):
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
        fn = lambda _kp_masks: destination_call(_kp_masks[0], _kp_masks[1], delta)
        maps = tf.vectorized_map(
            fn, 
            [kp, masks]
        )
        return maps

    def __build_heatmap(self, kp, masks, delta):
        """
        Builds a hard classification heatmap for a single keypoint `kp`.
        Parameters
        ----------
        kp : tf.Tensor of shape [2]
            A keypoint (x, y) for which to build the heatmap.
        masks : tf.Tensor of shape [1]
            Determines whether keypoint exists.
        xy_grid : tf.Tensor of shape [h, w, 2]
            A coordinate grid for the image tensor.
        delta : tf.float32
            Radius of the classification (heat) region.
        """
        xy_grid = self.xy_grid

        heatmap = tf.exp(
            -((xy_grid[..., 0] - kp[0])**2 + (xy_grid[..., 1] - kp[1])**2) / delta**2
        )
        return heatmap * masks


class PAFLayer(MakiLayer):
    IM_SIZE = 'im_size'
    SIGMA = 'sigma'
    SKELETON = 'skeleton'
    VECTORIZE = 'vectorize'

    RESIZE_TO = 'resize_to'
    PAF_RESIZE = 'paf_resize'

    @staticmethod
    def build(params: dict):
        return PAFLayer(
            im_size=params[PAFLayer.IM_SIZE],
            sigma=params[PAFLayer.SIGMA],
            skeleton=params[PAFLayer.SKELETON],
            vectorize=params[PAFLayer.VECTORIZE],
            resize_to=params[PAFLayer.RESIZE_TO]
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
        # Gather points along the axis of classes of points.
        # [b, n_pairs, 2, p, 2]
        kp_p = tf.gather(kp, indices=self.skeleton, axis=1)
        # This is needed for proper matrix multiplication during paf generation.
        kp_p = tf.transpose(kp_p, perm=[0, 1, 3, 2, 4])
        kp_p = tf.expand_dims(kp_p, axis=-1)
        # [b, n_pairs, p, 2, 2, 1]
        assert len(kp_p.get_shape()) == 6, f'Expected keypoint pairs dimensionality to be 6, but got {len(kp_p.get_shape())}.' + \
                f'Keypoints shape: {kp_p.get_shape()}'

        # Select masks for corresponding points.
        # [b, n_pairs, 2, p, 1]
        masks_p = tf.gather(masks, indices=self.skeleton, axis=1)
        masks_p = tf.transpose(masks_p, perm=[0, 1, 3, 2, 4])
        # [h, w, 2]
        fn_p = lambda kp, masks: tf.reduce_sum(
            PAFLayer.__build_paf_mp(
                kp, masks,
                destination_call=self.__build_paf
            ),
            axis=0
        )

        # [n_pairs, h, w, 2]
        fn_np = lambda kp, masks: PAFLayer.__build_paf_mp(
                kp, masks,
                destination_call=fn_p
        )

        # [b, n_pairs, h, w, 2]
        fn_b = lambda kp, masks: PAFLayer.__build_paf_mp(
                kp, masks,
                destination_call=fn_np
        )
        # Decide whether to perform calculation in a batch dimension.
        # May be faster, but requires more memory.
        if self.vectorize:            # [b, c, p, 2]
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
                [kp_p, masks_p]
            )
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
        # Define the required variables.
        p1 = p1p2[0]
        p2 = p1p2[1]
        h, w, _ = self.xy_grid.get_shape()
        # Flatten the field. It is needed for the later matrix multiplication.
        xy_flat = tf.reshape(self.xy_grid, [-1, 2])
        l = tf.maximum(tf.linalg.norm(p2 - p1), 1e-5)
        v = (p2 - p1) / l
        v_orth = tf.matmul(PAFLayer.ROT90_MAT, v)

        # Generate a mask for the points between `p1` and `p2`.
        c1 = tf.cast(0. <= tf.matmul(xy_flat - p1[:,0], v), tf.float32)
        c2 = tf.cast(tf.matmul(xy_flat - p1[:,0], v) <= l, tf.float32)
        cf_l = c1 * c2
        cf_l = tf.reshape(cf_l, [h, w])
        
        # Build a mask for the points lying on the line connecting `p1` and `p2`
        # with the width of `sigma`.
        cf_sigma = tf.abs(tf.matmul(xy_flat - p1[:, 0], v_orth)) <= self.sigma
        cf_sigma = tf.cast(cf_sigma, tf.float32)
        cf_sigma = tf.reshape(cf_sigma, [h, w])
        
        cf = cf_l * cf_sigma
        # Multiply the mask with the direction vector.
        paf = tf.expand_dims(cf, axis=-1) * v[:, 0]
        return paf * tf.reduce_min(points_mask)
