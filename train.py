import argparse
from itertools import chain
import random
import threading
from dataset import *
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')

parser.add_argument('--load_epoch', type=str, default='-1', help='Set load epoch')
parser.add_argument('--model_path', type=str, default='./models/', help='Set path for trained models')

parser.add_argument('--result_path', type=str, default='./result/', help='Set the path the result images will be saved.')
parser.add_argument('--input_path', type=str, default='./input/', help='Set the path the input images will be test.')

parser.add_argument('--save_iter', type=int, default=1000, help='Set save iter')

parser.add_argument('--epoch', type=int, default=5000, help='Set epoch')
parser.add_argument('--image_size', type=int, default=64, help='Set image size')
parser.add_argument('--test_size', type=int, default=200, help='Set test size')
parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')

parser.add_argument('--learning_rate', type=float, default=0.0002, help='Set learning rate')

def save_iamge_model(n_iter, generator_A, generator_B, discriminator_A, discriminator_B, test_A, test_B):
    print("save image[", n_iter, "]")

    save_path = os.path.join(args.result_path, str(n_iter))

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for i in range(0, args.test_size):
        AB = generator_B(test_A)
        BA = generator_A(test_B)
        ABA = generator_A(AB)
        BAB = generator_B(BA)

        save_image(str(i) + "_A", test_A[i], save_path)
        save_image(str(i) + "_B", test_B[i], save_path)
        save_image(str(i) + "_AB", AB[i], save_path)
        save_image(str(i) + "_BA", BA[i], save_path)
        save_image(str(i) + "_ABA", ABA[i], save_path)
        save_image(str(i) + "_BAB", BAB[i], save_path)

    torch.save(generator_A, os.path.join(args.model_path, 'model_gen_A-' + str(n_iter)))
    torch.save(generator_B, os.path.join(args.model_path, 'model_gen_B-' + str(n_iter)))
    torch.save(discriminator_A, os.path.join(args.model_path, 'model_dis_A-' + str(n_iter)))
    torch.save(discriminator_B, os.path.join(args.model_path, 'model_dis_B-' + str(n_iter)))


def get_gan_loss(dis_real, dis_fake, criterion):
    lables_dis_real = Variable(torch.ones([dis_real.size()[0], 1]))
    lables_dis_fake = Variable(torch.zeros([dis_fake.size()[0], 1]))
    lables_gen = Variable(torch.ones([dis_fake.size()[0], 1]))

    if torch.cuda:
        lables_dis_real = lables_dis_real.cuda()
        lables_dis_fake = lables_dis_fake.cuda()
        lables_gen = lables_gen.cuda()

    dis_loss = criterion(dis_real, lables_dis_real) * 0.5 + criterion(dis_fake, lables_dis_fake) * 0.5
    gen_loss = criterion(dis_fake, lables_gen)

    return dis_loss, gen_loss


def get_fm_loss(real_feats, fake_feats, criterion):
    losses = 0

    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        l2_real = Variable(torch.ones(l2.size()))

        if torch.cuda:
            l2_real = l2_real.cuda()

        loss = criterion(l2, l2_real)
        losses += loss

    return losses


def main():
    global args
    args = parser.parse_args()

    if not os.path.isdir(args.result_path):
        os.mkdir(args.result_path)
    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)

    generator_A, generator_B, discriminator_A, discriminator_B = get_model(args.model_path, args.load_epoch)

    #test_A = get_real_image(args.image_size, os.path.join(input_path, 'A'), True, args.test_size)
    #test_B = get_real_image(args.image_size, os.path.join(input_path, 'B'), True, args.test_size)
    #train_A = get_real_image(args.image_size, os.path.join(input_path, 'A'), False, args.test_size)
    #train_B = get_real_image(args.image_size, os.path.join(input_path, 'B'), False, args.test_size)
    test_A, train_A = get_real_image(args.image_size, os.path.join(args.input_path, 'A'), args.test_size)
    test_B, train_B = get_real_image(args.image_size, os.path.join(args.input_path, 'B'), args.test_size)

    test_A = Variable(torch.FloatTensor(test_A))
    test_B = Variable(torch.FloatTensor(test_B))


    if torch.cuda:
        test_A = test_A.cuda()
        test_B = test_B.cuda()


    #need to add batch
    data_size = min(len(train_A), len(train_B))
    n_batchs = data_size // args.batch_size


    recon_crierion = nn.MSELoss()
    gan_crierion = nn.BCELoss()
    feat_crierion = nn.HingeEmbeddingLoss()

    gen_params = chain(generator_A.parameters(), generator_B.parameters())
    dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())

    optim_gen = torch.optim.Adam(gen_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    optim_dis = torch.optim.Adam(dis_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)

    iter = 0
    for epoch in range(args.epoch):
        print("epoch :", epoch)
        #training

        random.shuffle(train_A)
        random.shuffle(train_B)

        generator_A.zero_grad()
        generator_B.zero_grad()
        discriminator_A.zero_grad()
        discriminator_B.zero_grad()

        for i in range(n_batchs):
            A = train_A[n_batchs * i:n_batchs * (i + 1)]
            B = train_B[n_batchs * i:n_batchs * (i + 1)]

            A = Variable(torch.FloatTensor(A))
            B = Variable(torch.FloatTensor(B))

            if torch.cuda:
                A = A.cuda()
                B = B.cuda()

            AB = generator_B(A)
            BA = generator_A(B)
            ABA = generator_A(AB)
            BAB = generator_B(BA)

            # loss

            #reconstruction loss
            recon_loss_A = recon_crierion(ABA,A)
            recon_loss_B = recon_crierion(BAB,B)

            #gan loss
            A_dis_real, A_feats_real = discriminator_A(A, args.image_size)
            A_dis_fake, A_feats_fake = discriminator_A(BA, args.image_size)

            dis_loss_A, gen_loss_A = get_gan_loss(A_dis_real, A_dis_fake, gan_crierion)
            fm_loss_A = get_fm_loss(A_feats_real, A_feats_fake, feat_crierion)

            B_dis_real, B_feats_real = discriminator_B(B, args.image_size)
            B_dis_fake, B_feats_fake = discriminator_B(AB, args.image_size)

            dis_loss_B, gen_loss_B = get_gan_loss(B_dis_real, B_dis_fake, gan_crierion)
            fm_loss_B = get_fm_loss(B_feats_real, B_feats_fake, feat_crierion)


            #total loss
            gen_loss_A_total = (gen_loss_B * 0.1 + fm_loss_B * 0.9) * 0.5 + recon_loss_A * 0.5
            gen_loss_B_total = (gen_loss_A * 0.1 + fm_loss_A * 0.9) * 0.5 + recon_loss_B * 0.5

            gen_loss = gen_loss_A_total + gen_loss_B_total
            dis_loss = dis_loss_A + dis_loss_B

            if iter % 3 == 0:
                dis_loss.backward()
                optim_dis.step()
            else:
                gen_loss.backward()
                optim_gen.step()

            #save validation set
            if iter % args.save_iter == 0:
                t = threading.Thread\
                    (target=save_iamge_model,
                     args=(iter // args.save_iter, generator_A, generator_B, discriminator_A, discriminator_B, test_A, test_B))
                t.start()

            iter = iter + 1

if __name__=="__main__":
    main()