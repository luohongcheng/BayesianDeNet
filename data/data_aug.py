import os
import cv2
import random


def scale_transform(image, scale_ratio):
    new_width = int(scale_ratio * 640)
    new_height = int(scale_ratio * 480)
    image_result = cv2.resize(image, (new_width, new_height))
    image_result = image_result[new_height / 2 - 240:new_height / 2 + 240, new_width / 2 - 320:new_width / 2 + 320]
    return image_result


def color_transform(image, scale_ratio):
    image_result = image * scale_ratio
    image_result[image_result > 255] = 255
    image_result[image_result < 0] = 0
    return image_result


def flip_transfrom(image, id):
    if id == 2:
        return image
    image_result = cv2.flip(image, id)
    return image_result


def listdir(path):
    file_list = []
    for file in os.listdir(path):
        file_list.append(file)
    return file_list


if __name__ == '__main__':
    input_dir = "/home/dataset/nyu_dataset/"
    output_dir = "/home/dataset/nyu_dataset_augmented/"
    file_list = listdir(input_dir + "rgb/")

    image_count = 0

    for file_name in file_list:
        rgb_image = cv2.imread(input_dir + "rgb/" + file_name, 1)
        depth_image = cv2.imread(input_dir + "depth/" + file_name, -1)
        print input_dir + "rgb/" + file_name
        scale_ratio = random.uniform(1, 1.5)
        color_ratio = random.uniform(0.6, 1.4)

        # 2^3=8
        for flip in [1, 2]:
            flipped_rgb_image = flip_transfrom(rgb_image, flip)
            flipped_depth_image = flip_transfrom(depth_image, flip)

            for color in [1, color_ratio]:
                colored_rgb_image = color_transform(flipped_rgb_image, color)  # .astype('uint8')
                colored_depth_image = flipped_depth_image

                for scale in [1, scale_ratio]:
                    scaled_rgb_image = scale_transform(colored_rgb_image, scale);
                    scaled_depth_image = scale_transform(colored_depth_image, scale);
                    scaled_depth_image = (scaled_depth_image / scale).astype('uint16')
                    cv2.imwrite(output_dir + "rgb/%08d.png" % image_count, scaled_rgb_image)
                    cv2.imwrite(output_dir + "depth/%08d.png" % image_count, scaled_depth_image)
                    image_count += 1
