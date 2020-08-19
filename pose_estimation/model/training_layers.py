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



