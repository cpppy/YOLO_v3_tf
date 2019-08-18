


def calc_rescale_size(img_shape, max_size=400):
    img_w = img_shape[1]
    img_h = img_shape[0]
    if img_w >= img_h:
        rescaled_w = max_size
        rescaled_h = int(img_h * rescaled_w / img_w)
        rescale_ratio = rescaled_w / img_w

    else:
        rescaled_h = max_size
        rescaled_w = int(img_w * rescaled_h / img_h)
        rescale_ratio = rescaled_h / img_h
    return (rescaled_w, rescaled_h), rescale_ratio
