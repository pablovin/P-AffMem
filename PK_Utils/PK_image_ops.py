from __future__ import division
import numpy as np
import cv2

def load_image(
        image_path,  # path of a image
        image_size=64,  # expected size of the image
        image_value_range=(-1, 1),  # expected pixel value range of the image
        is_gray=False,  # gray scale or color image
):
    if is_gray:
        # image = imread(image_path, flatten=True).astype(np.float32)
        image = cv2.imread(image_path, 0).astype(np.float32)
    else:
        # image = imread(image_path).astype(np.float32)
        image = cv2.imread(image_path)

    # image = imresize(image, [image_size, image_size])
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
    # image = image.astype(np.float32)/255.0
    return image


def save_batch_images(
        batch_images,   # a batch of images
        save_path,  # path to save the images
        image_value_range=(-1,1),   # value range of the input batch images
        size_frame=None     # size of the image matrix, number of images in each row and column
):
    # transform from 0~1 to pixel value
    images = (batch_images - image_value_range[0]) / (image_value_range[-1] - image_value_range[0])
    images = images*255
    # images = (batch_images + 1) / 2 * 255
    if size_frame is None:
        auto_size = int(np.ceil(np.sqrt(images.shape[0])))
        size_frame = [auto_size, auto_size]
    img_h, img_w = batch_images.shape[1], batch_images.shape[2]
    frame = np.zeros([img_h * size_frame[0], img_w * size_frame[1], 3])

    # print ("Frame:" + str(frame.shape))
    for ind, image in enumerate(images):
        ind_col = ind % size_frame[1]
        ind_row = ind // size_frame[1]
        # print ("From: " + str(ind_row * img_h)+":"+str(ind_row * img_h + img_h))
        # print ("to: " + str(ind_col * img_w) + ":" + str(ind_col * img_w + img_w) )
        frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w), :] = image
    cv2.imwrite(save_path, frame)

def save_output(input_image, output, path, image_value_range = (-1,1), size_frame=[6, 7]):

    # Tile black background
    black_image = np.zeros((1, 96, 96, 3))
    black_image2 = np.tile(black_image, (2, 1, 1, 1))

    # Build final image from components
    # input_frame = np.concatenate([black_image2,output[:4],black_image, input_image, black_image,output[4:8],black_image3,output[8:12],black_image3,output[12:]])

    input_frame = np.concatenate([black_image2,output[:5],
                                  black_image2, output[5:10],
                                  input_image,  black_image,output[10:15],
                                  black_image2,output[15:20],
                                  black_image2, output[20:25],
                                  ])


    # Transform into savable format
    final_image = get_images_frame(input_frame, image_value_range=image_value_range, size_frame=size_frame)

    # Save image
    cv2.imwrite(path, final_image)

def get_images_frame(
        batch_images,  # a batch of images
        image_value_range=(-1, 1),  # value range of the input batch images
        size_frame=None  # size of the image matrix, number of images in each row and column
):
    # transform the pixcel value to 0~1
    images = (batch_images - image_value_range[0]) / (image_value_range[-1] - image_value_range[0])
    images = images * 255
    # images = (batch_images + 1) / 2 * 255
    if size_frame is None:
        auto_size = int(np.ceil(np.sqrt(images.shape[0])))
        size_frame = [auto_size, auto_size]
    img_h, img_w = batch_images.shape[1], batch_images.shape[2]
    frame = np.zeros([img_h * size_frame[0], img_w * size_frame[1], 3])
    for ind, image in enumerate(images):
        ind_col = ind % size_frame[1]
        ind_row = ind // size_frame[1]
        frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w), :] = image
    return frame


