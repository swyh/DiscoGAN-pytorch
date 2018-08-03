from torch.autograd import Variable
import torch

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