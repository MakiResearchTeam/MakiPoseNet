from makiflow.layers.sf_layer import SimpleForwardLayer
from makiflow.base.maki_entities.maki_layer import MakiRestorable
import tensorflow as tf
import numpy as np


class BinaryHeatmapLayer(SimpleForwardLayer):

    def __init__(self, im_size, radius, map_dtype=tf.int32, vectorize=False, name='BinaryHeatmapLayer'):
        """
        Generates hard keypoint maps using highly optimized vectorization.

        Parameters
        ----------
        im_size : 2d tuple
            Contains width and height (w, h) of the image for which to generate the map.
        radius : int
            Radius of a label-circle around the keypoint.
        map_dtype : tf.dtype
            Dtype of the generated map. Use tf.int32 for binary classification and tf.float32 for
            regression.
        vectorize : bool
            Set to True if you want to vectorize the computation along the batch dimension. May cause
            the OOM error due to high memory consumption.
        """
        super().__init__(name, params=[], regularize_params=[], named_params_dict={})
        self.im_size = im_size
        self.radius = tf.convert_to_tensor(radius, dtype=tf.float32)
        self.map_dtype = map_dtype
        self.vectorize = vectorize
        # Prepare the grid.
        x_grid, y_grid = np.meshgrid(np.arange(im_size[0]), np.arange(im_size[1]))
        xy_grid = np.stack([x_grid, y_grid], axis=-1)
        self.xy_grid = tf.convert_to_tensor(xy_grid, dtype=tf.float32)

    def _forward(self, x, computation_mode=MakiRestorable.TRAINING_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                keypoints = x
                maps = self.__build_heatmap_batch(keypoints, self.xy_grid, self.radius)
        return maps

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {}

    def __build_heatmap_batch(self, kp, xy_grid, radius):
        # Build maps for keypoints of the same class for multiple people
        # and then aggregate generated maps.
        # [h, w]
        fn_p = lambda kp, xy_grid, radius: tf.reduce_max(
            BinaryHeatmapLayer.__build_heatmap_mp(
                kp, 
                xy_grid, 
                radius, 
                destination_call=self.__build_heatmap
            ),
            axis=0
        )
        # Build maps for keypoints of multiple classes.
        # [c, h, w]
        fn_c = lambda kp, xy_grid, radius: BinaryHeatmapLayer.__build_heatmap_mp(
            kp, 
            xy_grid, 
            radius, 
            destination_call=fn_p
        )
        # Build a batch of maps.
        # [b, c, h, w]
        fn_b = lambda kp, xy_grid, radius: BinaryHeatmapLayer.__build_heatmap_mp(
            kp, 
            xy_grid, 
            radius, 
            destination_call=fn_c
        )
        
        # Decide whether to perform calucalation in a batch dimension.
        # May be faster, but requires more memory.
        if len(kp.get_shape()) == 4 and self.vectorize:            # [b, c, p, 2]
            print('Using vectorized_map.')
            return fn_b(kp, xy_grid, radius)
        elif len(kp.get_shape()) == 4 and not self.vectorize: 
            # Requires less memory, but runs slower
            print('Using map_fn.')
            fn = lambda kp_: fn_c(kp_, xy_grid, radius)
            return tf.map_fn(
                fn,
                kp
            )
        else:
            message = f'Expected keypoints dimensionality to be 4, but got {len(kp.get_shape())}.' + \
                f'Keypoints shape: {kp.get_shape()}'
            raise Exception(message)

    @staticmethod
    def __build_heatmap_mp(kp, xy_grid, radius, destination_call):
        """
        The hetmaps generation is factorized down to single keypoint heatmap generation.
        Nested calls of this method allow for highly optimized vectorized computation of multiple maps.

        Parameters
        ----------
        kp : tf.Tensor of shape [..., 2]
            A keypoint (x, y) for which to build the heatmap.
        xy_grid : tf.Tensor of shape [h, w, 2]
            A coordinate grid for the image tensor.
        radius : tf.float32
            Radius of the classification (heat) region.
        destination_call : method pointer
            Used for nested calling to increase the dimensionality of the computation.
        """
        fn = lambda _kp: destination_call(_kp, xy_grid, radius)
        maps = tf.vectorized_map(fn, kp)
        return maps

    def __build_heatmap(self, kp, xy_grid, radius):
        """
        Builds a hard classification heatmap for a single keypoint `kp`.
        Parameters
        ----------
        kp : tf.Tensor of shape [2]
            A keypoint (x, y) for which to build the heatmap.
        xy_grid : tf.Tensor of shape [h, w, 2]
            A coordinate grid for the image tensor.
        radius : tf.float32
            Radius of the classification (heat) region.
        """
        print(kp.get_shape())
        grid_size = xy_grid.get_shape()[:2]
        heatmap = tf.ones((grid_size[0], grid_size[1]), dtype=self.map_dtype)

        bool_location_map = (xy_grid[..., 0] - kp[0])**2 + (xy_grid[..., 1] - kp[1])**2 < radius**2
        bool_location_map = tf.cast(bool_location_map, dtype=self.map_dtype)
        return heatmap * bool_location_map


class PAFLayer(SimpleForwardLayer):
    # 90 degrees rotation matrix. Used for generation of orthogonal vector. 
    ROT90_MAT = tf.convert_to_tensor(
        np.array([
            [0, 1],
            [-1, 0]
        ]),
        dtype=tf.float32
    )

    def __init__(self, im_size, sigma, skeleton, vectorize=False, name='PAFLayer'):
        """
        Generates part affinity fields for the given `skeleton`.

        Parameters
        ----------
        im_size : 2d tuple
            Contains width and height (w, h) of the image for which to generate the map.
        sigma : int
            Width of the affinity field. Corresponds to the width of the limb.
        skeleton : np.ndarray of shape [n_pairs, 2]
            A numpy array containing indices for pairs of points. Vectors in the PAF
            will be equal the following vector:
            (point1 - point0) / norm(point1 - point0).
        vectorize : bool
            Set to True if you want to vectorize the computation along the batch dimension. May cause
            the OOM error due to high memory consumption.
        """
        super().__init__(name, params=[], regularize_params=[], named_params_dict={})
        self.sigma = sigma
        self.skeleton = skeleton
        self.im_size = im_size
        self.vectorize = vectorize
        x_grid, y_grid = np.meshgrid(np.arange(im_size[0]), np.arange(im_size[1]))
        xy_grid = np.stack([x_grid, y_grid], axis=-1)
        self.xy_grid = tf.convert_to_tensor(xy_grid, dtype=tf.float32)

    def _forward(self, x, computation_mode=MakiRestorable.TRAINING_MODE):
        with tf.name_scope(computation_mode):
            with tf.name_scope(self.get_name()):
                keypoints = x
                pafs = self.__build_paf_batch(keypoints)
        return pafs

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {}   

    def __build_paf_batch(self, kp):
        """
        Generates PAF for the given keypoints.

        Parameters
        ----------
        kp : tf.Tensor of shape [batch, n_people, c, 2]
            Tensor of keypoints coordinates.
        
        Returns
        -------
        tf.Tensor of shape [batch, n_pairs, h, w]
            Tensor of PAFs.
        """
        # Gather points along the axis of classes of points.
        # [b, p, n_pairs, 2, 2]
        kp_p = tf.gather(kp, indices=self.skeleton, axis=2)
        print(kp.get_shape())
        # [b, n_pairs, p, 2, 2]
        kp_p = tf.transpose(kp_p, perm=[0, 2, 1, 3, 4])
        # This is needed for proper matrix multiplication during paf generation.
        kp_p = tf.expand_dims(kp_p, axis=-1)
        print(kp_p.get_shape())
        assert len(kp_p.get_shape()) == 6, f'Expected keypoint pairs dimensionality to be 6, but got {len(kp_p.get_shape())}.' + \
                f'Keypoints shape: {kp_p.get_shape()}'

        # [h, w, 2]
        fn_p = lambda kp: tf.reduce_mean(
            PAFLayer.__build_paf_mp(
                kp, 
                destination_call=self.__build_paf
            ),
            axis=0
        )

        # [n_pairs, h, w, 2]
        fn_np = lambda kp: PAFLayer.__build_paf_mp(
                kp, 
                destination_call=fn_p
        )

        # [b, n_pairs, h, w, 2]
        fn_b = lambda kp: PAFLayer.__build_paf_mp(
                kp, 
                destination_call=fn_np
        )
        # Decide whether to perform calucalation in a batch dimension.
        # May be faster, but requires more memory.
        if self.vectorize:            # [b, c, p, 2]
            print('Using vectorized_map.')
            return fn_b(kp_p)
        else: 
            # Requires less memory, but runs slower
            print('Using map_fn.')
            fn = lambda kp_: fn_np(kp_)
            return tf.map_fn(
                fn,
                kp_p
            )

    @staticmethod
    def __build_paf_mp(kp, destination_call):
        """
        The hetmaps generation is factorized down to single keypoint heatmap generation.
        Nested calls of this method allow for highly optimized vectorized computation of multiple maps.

        Parameters
        ----------
        kp : tf.Tensor of shape [..., 2]
            A keypoint (x, y) for which to build the heatmap.
        xy_grid : tf.Tensor of shape [h, w, 2]
            A coordinate grid for the image tensor.
        radius : tf.float32
            Radius of the classification (heat) region.
        destination_call : method pointer
            Used for nested calling to increase the dimensionality of the computation.
        """
        print('mp', kp.get_shape())
        fn = lambda _kp: destination_call(_kp)
        maps = tf.vectorized_map(fn, kp)
        return maps

    def __build_paf(self, p1p2):
        """
        Build PAF for the given 2 points `p1p2` on the `xy_grid`.
        
        Parameters
        ----------
        p1p2 : tf.Tensor of shape [2, 2, 1]
            Points between which to generate the field. Vectors will points from
            p1 to p2.
        sigma : tf.float32
            Width of the field.
        xy_grid : tf.Tensor of shape [h, w, 2]
            A coordinate grid of the image plane.
        
        Returns
        -------
        tf.Tensor of shape [h, w, 2]
            The generated PAF.
        """
        print(p1p2.get_shape())
        # Define the required variables.
        p1 = p1p2[0]
        p2 = p1p2[1]
        h, w, _ = self.xy_grid.get_shape()
        # Flatten the field. It is needed for the later matrix multiplication.
        xy_flat = tf.reshape(self.xy_grid, [-1, 2])
        l = tf.linalg.norm(p2 - p1)
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
        # Mutiply the mask with the direction vector.
        paf = tf.expand_dims(cf, axis=-1) * v[:, 0]
        return paf