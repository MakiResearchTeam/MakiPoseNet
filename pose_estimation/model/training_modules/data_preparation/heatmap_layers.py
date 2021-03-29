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
        map_dtype = params.get(BinaryHeatmapLayer.MAP_DTYPE)
        if map_dtype is None:
            map_dtype = tf.float32

        return BinaryHeatmapLayer(
            im_size=params[BinaryHeatmapLayer.IM_SIZE],
            delta=params[BinaryHeatmapLayer.DELTA],
            map_dtype=map_dtype,
            vectorize=params[BinaryHeatmapLayer.VECTORIZE],
            resize_to=params[BinaryHeatmapLayer.RESIZE_TO]
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
        if len(kp.get_shape()) == 4 and self.vectorize:  # [b, c, p, 2]
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

        bool_location_map = (xy_grid[..., 0] - kp[0]) ** 2 + (xy_grid[..., 1] - kp[1]) ** 2 < delta ** 2
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
            resize_to=params[GaussHeatmapLayer.RESIZE_TO]
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

        # Decide whether to perform calculation in a batch dimension.
        # May be faster, but requires more memory.
        if len(kp.get_shape()) == 4 and self.vectorize:  # [b, c, p, 2]
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
            -((xy_grid[..., 0] - kp[0]) ** 2 + (xy_grid[..., 1] - kp[1]) ** 2) / delta ** 2
        )
        return heatmap * masks

