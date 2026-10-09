# from https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py
import math
import random
import numpy as np
from PIL import Image


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def random_crop_arr2(pil_image, image_width, image_height, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_width = math.ceil(image_width / max_crop_frac) # 256 480
    max_smaller_dim_width = math.ceil(image_width / min_crop_frac) # 320 600
    smaller_dim_width = random.randrange(min_smaller_dim_width, max_smaller_dim_width + 1) # 270 520
    min_smaller_dim_height = math.ceil(image_height / max_crop_frac) # 256 384
    max_smaller_dim_height = math.ceil(image_height / min_crop_frac) # 320 480
    smaller_dim_height = random.randrange(min_smaller_dim_height, max_smaller_dim_height + 1) # 270 411
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while pil_image.size[0] >= 2 * smaller_dim_width and pil_image.size[1] >= 2 * smaller_dim_height: # 540 1040/822
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale_width = smaller_dim_width / pil_image.size[0]
    scale_height = smaller_dim_height / pil_image.size[1]
    scale = max(scale_width, scale_height)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_height + 1)
    crop_x = random.randrange(arr.shape[1] - image_width + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_height, crop_x : crop_x + image_width])

def random_crop_arr3(pil_image, image_width, image_height, min_crop_frac=0.8, max_crop_frac=1.0):
    # min_smaller_dim_width = math.ceil(image_width / max_crop_frac) # 256 480
    # max_smaller_dim_width = math.ceil(image_width / min_crop_frac) # 320 600
    # smaller_dim_width = random.randrange(min_smaller_dim_width, max_smaller_dim_width + 1) # 270 520
    # min_smaller_dim_height = math.ceil(image_height / max_crop_frac) # 256 384
    # max_smaller_dim_height = math.ceil(image_height / min_crop_frac) # 320 480
    # smaller_dim_height = random.randrange(min_smaller_dim_height, max_smaller_dim_height + 1) # 270 411
    # # We are not on a new enough PIL to support the `reducing_gap`
    # # argument, which uses BOX downsampling at powers of two first.
    # # Thus, we do it by hand to improve downsample quality.
    # while pil_image.size[0] >= 2 * smaller_dim_width and pil_image.size[1] >= 2 * smaller_dim_height: # 540 1040/822
    #     pil_image = pil_image.resize(
    #         tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        # )

    scale_width =  pil_image.size[0] / image_width #  4000 512 7.8
    scale_height =  pil_image.size[1] / image_height #  3000 256 11.7
    scale = int(min(scale_width, scale_height)) # 7 3584 1792
    if scale == 0:
        if scale_width < scale_height:
            new_width = pil_image.size[0]
            new_height = int(pil_image.size[1] * scale_width / scale_height)
        else:
            new_width = int(pil_image.size[0] / scale_width * scale_height)
            new_height = pil_image.size[1]
    else:
        new_width = image_width * scale
        new_height = image_height * scale


    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - new_height + 1)
    crop_x = random.randrange(arr.shape[1] - new_width + 1)
    croped_pil = Image.fromarray(arr[crop_y : crop_y + new_height, crop_x : crop_x + new_width])
    resized_pil = croped_pil.resize(
        (image_width, image_height), resample=Image.BICUBIC
    )
    return resized_pil

def random_crop_arr4(pil_image, image_width, image_height, min_crop_frac=0.8, max_crop_frac=1.0, scalex=0.5, scaley=0.5):
    # min_smaller_dim_width = math.ceil(image_width / max_crop_frac) # 256 480
    # max_smaller_dim_width = math.ceil(image_width / min_crop_frac) # 320 600
    # smaller_dim_width = random.randrange(min_smaller_dim_width, max_smaller_dim_width + 1) # 270 520
    # min_smaller_dim_height = math.ceil(image_height / max_crop_frac) # 256 384
    # max_smaller_dim_height = math.ceil(image_height / min_crop_frac) # 320 480
    # smaller_dim_height = random.randrange(min_smaller_dim_height, max_smaller_dim_height + 1) # 270 411
    # # We are not on a new enough PIL to support the `reducing_gap`
    # # argument, which uses BOX downsampling at powers of two first.
    # # Thus, we do it by hand to improve downsample quality.
    # while pil_image.size[0] >= 2 * smaller_dim_width and pil_image.size[1] >= 2 * smaller_dim_height: # 540 1040/822
    #     pil_image = pil_image.resize(
    #         tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        # )

    scale_width =  pil_image.size[0] / image_width #  4000 512 7.8
    scale_height =  pil_image.size[1] / image_height #  3000 256 11.7
    scale = int(min(scale_width, scale_height)) # 7 3584 1792
    if scale == 0:
        if scale_width < scale_height:
            new_width = pil_image.size[0]
            new_height = int(pil_image.size[1] * scale_width / scale_height)
        else:
            new_width = int(pil_image.size[0] / scale_width * scale_height)
            new_height = pil_image.size[1]
    else:
        new_width = image_width * scale
        new_height = image_height * scale


    arr = np.array(pil_image)
    crop_y = int((arr.shape[0] - new_height + 1)*scaley)
    crop_x = int((arr.shape[1] - new_width + 1)*scalex)
    croped_pil = Image.fromarray(arr[crop_y : crop_y + new_height, crop_x : crop_x + new_width])
    resized_pil = croped_pil.resize(
        (image_width, image_height), resample=Image.BICUBIC
    )
    return resized_pil

def val_resize_arr(pil_image, image_width, image_height, min_crop_frac=0.8, max_crop_frac=1.0):
    return pil_image.resize((image_width, image_height), Image.BICUBIC)

def np_resize_arr(np_image, image_width, image_height):
    return np.array(Image.fromarray(np_image).resize((image_width, image_height), Image.BICUBIC))/255.*2-1