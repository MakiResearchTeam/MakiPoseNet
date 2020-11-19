

def scale_predicted_kp(predictions: list, model_size: tuple, source_size: tuple):
    # scale predictions
    scale = [source_size[1] / model_size[1], source_size[0] / model_size[0]]

    # each detection
    for i in range(len(predictions)):
        single_image_pr = predictions[i]
        for h_indx in range(len(single_image_pr)):
            single_human = single_image_pr[h_indx]
            for kp_indx in single_human.body_parts:
                single_human.body_parts[kp_indx].x *= scale[0]
                single_human.body_parts[kp_indx].y *= scale[1]

    return predictions

