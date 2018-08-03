import torch
import os
import numpy as np
from scipy import misc
from torch.autograd import Variable
from option import TestOption
from dataset import *



def main():
    args = TestOption().parse()

    generator_A, generator_B, discriminator_A, discriminator_B = get_model(args.model_path, args.load_epoch)

    test_A1, test_A2 = get_real_image(args.image_size, os.path.join(args.input_path, 'A'))
    test_B1, test_B2 = get_real_image(args.image_size, os.path.join(args.input_path, 'B'))

    test_A = test_A2 # test_A1 : zero
    test_B = test_B2

    A = Variable(torch.FloatTensor(test_A))
    B = Variable(torch.FloatTensor(test_B))

    if torch.cuda:
        A = A.cuda()
        B = B.cuda()

    save_all_image(args.result_path, len(A), generator_A, generator_B, A, B)


if __name__=="__main__":
    main()
