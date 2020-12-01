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

def scale_predicted_kp(predictions: list, model_size: tuple, source_size: tuple):
    # predictions shape - (N, num_detected_people)
    # scale predictions
    scale = [source_size[1] / model_size[1], source_size[0] / model_size[0]]

    # each detection
    for i in range(len(predictions)):
        single_image_pr = predictions[i]
        # each person
        for h_indx in range(len(single_image_pr)):
            single_human = single_image_pr[h_indx]
            # each keypoint
            for kp_indx in single_human.body_parts:
                single_human.body_parts[kp_indx].x *= scale[0]
                single_human.body_parts[kp_indx].y *= scale[1]

    return predictions

