import argparse

class Option():

    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')

        parser.add_argument('--load_epoch', type=str, default='-1', help='Set load epoch')
        parser.add_argument('--model_path', type=str, default='./models/', help='Set generated model path')
        parser.add_argument('--result_path', type=str, default='./result/', help='Set the result images path')
        parser.add_argument('--input_path', type=str, default='./test/', help='Set the input images path')


        parser.add_argument('--epoch', type=int, default=5000, help='Set epoch')
        parser.add_argument('--image_size', type=int, default=64, help='Set image size')
        parser.add_argument('--test_size', type=int, default=200, help='Set test size')
        parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')

        parser.add_argument('--learning_rate', type=float, default=0.0002, help='Set learning rate')
        parser.add_argument('--save_iter', type=int, default=1000, help='Set save iter')

        self.parser = parser


    def parse(self):
        return self.parser.parse_args()
