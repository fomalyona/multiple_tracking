import cv2 as cv
import os
import glob
import torch











#
# PATH_IMG = '/home/alena/PycharmProjects/TRACKER2/DATA/images'
#
#
# def read_img(path_file):
#     images = []
#     list_of_files = len(os.listdir(path_file))
#     for filename in range(1, list_of_files + 1):
#         with open(os.path.join(path_file, '0{}.png'.format(filename))) as img:
#             image = cv.imread(os.path.join(path_file, '0{}.png'.format(filename)), cv.IMREAD_COLOR)
#             images.append(image)
#     return images
#
# def show_result(images, PATH_OUTPUT):
#     h, w, _ = images[0].shape
#     out = cv.VideoWriter(PATH_OUTPUT, cv.VideoWriter_fourcc(*'MJPG'), 20.0, (w, h))
#     for i in range(len(images)):
#         cv.imshow('', images[i])
#         cv.waitKey(100)
#         out.write(images[i])
#     cv.destroyAllWindows()
#     out.release()
#
#
#
# images = read_img(PATH_IMG)
# show_result(images, '/home/alena/PycharmProjects/TRACKER2/DATA/LL_men.avi')