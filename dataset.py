import os
import cv2
import numpy as np
from scipy import misc
from model import *
import torch

def get_model(model_path, epoch):
    file_names = ['model_gen_A', 'model_gen_B', 'model_dis_A', 'model_dis_B']

    try:
        file_list = os.listdir(model_path)
        for file in file_names:
            if (file+epoch) in file_list:
                pass
            else:
                raise Exception('This is the exception you expect to handle')

        generator_A = torch.load(os.path.join(model_path, 'model_gen_A') + epoch)
        generator_B = torch.load(os.path.join(model_path, 'model_gen_B') + epoch)
        discriminator_A = torch.load(os.path.join(model_path, 'model_dis_A') + epoch)
        discriminator_B = torch.load(os.path.join(model_path, 'model_dis_B') + epoch)

        print("success to load models")
    except Exception as e:
        print(e)
        print("So, create new model")
        generator_A = Generator()
        generator_B = Generator()
        discriminator_A = Discriminator()
        discriminator_B = Discriminator()

    if torch.cuda:
        generator_A = generator_A.cuda()
        generator_B = generator_B.cuda()
        discriminator_A = discriminator_A.cuda()
        discriminator_B = discriminator_B.cuda()

    return generator_A, generator_B, discriminator_A, discriminator_B


def get_real_image(image_size=64, input_path="", test=False, test_size=200): # path 불러오기
    images = []

    file_list = os.listdir(input_path)

    for file in file_list:
        if file.endswith(".jpg") or file.endswith(".png"):
            file_path = os.path.join(input_path, file)

            image = cv2.imread(file_path)  # fn이 한글 경로가 포함되어 있으면 제대로 읽지 못함. binary로 바꿔서 처리하는 방법있음

            if image is None:
                print("None")
                continue
            # image를 image_size(default=64)로 변환
            image = cv2.resize(image, (image_size, image_size))
            image = image.astype(np.float32) / 255.
            image = image.transpose(2, 0, 1)
            images.append(image)

    if images:
        print("push the stack")
        images = np.stack(images)
    else:
        print("error, images is emtpy")

    if test == True:
        return images[:test_size]
    else:
        return images[test_size:]

def get_real_image(image_size=64, input_path="", test_size=200): # path 불러오기
    images = []

    file_list = os.listdir(input_path)

    for file in file_list:
        if file.endswith(".jpg") or file.endswith(".png"):
            file_path = os.path.join(input_path, file)

            image = cv2.imread(file_path)  # fn이 한글 경로가 포함되어 있으면 제대로 읽지 못함. binary로 바꿔서 처리하는 방법있음

            if image is None:
                print("None")
                continue
            # image를 image_size(default=64)로 변환
            image = cv2.resize(image, (image_size, image_size))
            image = image.astype(np.float32) / 255.
            image = image.transpose(2, 0, 1)
            images.append(image)

    if images:
        print("push image in the stack")
        images = np.stack(images)
    else:
        print("error, images is emtpy")

    return images[:test_size], images[test_size:]


def save_image(name, image, result_path):
    image = image.cpu().data.numpy().transpose(1, 2, 0) * 255.
    misc.imsave(os.path.join(result_path, name + '.jpg'), image.astype(np.uint8)[:, :, ::-1])
