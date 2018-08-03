from itertools import chain
import random
from dataset import *
from option import TrainOption
from loss import *

def main():
    global args
    args = TrainOption().parse()
    print(args.result_path)
    if not os.path.isdir(args.result_path):
        os.mkdir(args.result_path)
    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)

    generator_A, generator_B, discriminator_A, discriminator_B = get_model(args.model_path, args.load_epoch)

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
            A_dis_real, A_feats_real = discriminator_A(A)
            A_dis_fake, A_feats_fake = discriminator_A(BA)

            dis_loss_A, gen_loss_A = get_gan_loss(A_dis_real, A_dis_fake, gan_crierion)
            fm_loss_A = get_fm_loss(A_feats_real, A_feats_fake, feat_crierion)

            B_dis_real, B_feats_real = discriminator_B(B)
            B_dis_fake, B_feats_fake = discriminator_B(AB)

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
                n_iter = iter // args.save_iter
                print("start to save image and model[", n_iter, "]")
                save_path = os.path.join(args.result_path, str(n_iter))

                save_all_image(save_path, args.test_size, generator_A, generator_B, test_A, test_B)
                save_model(args.model_path, n_iter, generator_A, generator_B, discriminator_A, discriminator_B)

            iter = iter + 1


if __name__=="__main__":
    main()