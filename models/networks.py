import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import functions


class Identity(nn.Module):
    def forward(self, x):
        return x




def get_scheduler(optimizer, opt):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        init.normal_(m.weight.data, 0.0, init_gain)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = Generator()
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = Discriminator()
    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        model = []
        model += [nn.Conv2d(3, 64, kernel_size=4, stride = 2, padding=1, bias=True), nn.BatchNorm2d(64), nn.ReLU(True)]
        model += [nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True), nn.BatchNorm2d(64),nn.ReLU(True)]
        model += [nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True), nn.BatchNorm2d(64),nn.ReLU(True)]
        model += ResnetBlock()
        model += ResnetBlock()
        model += ResnetBlock()

        model += [nn.MultiHeadDotProductAttention(), nn.Conv2d(64, 64, kernel_size=3, stride = 1, padding=1, bias=True)]
        model += [nn.MultiHeadDotProductAttention(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        model += [nn.MultiHeadDotProductAttention(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        model += [nn.MultiHeadDotProductAttention(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]


        model += ResnetBlock()
        model += ResnetBlock()
        model += ResnetBlock()

        model += [nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True), nn.BatchNorm2d(64),nn.ReLU(True)]
        model += [nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True), nn.BatchNorm2d(64),nn.ReLU(True)]
        model += [nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=True)]

        model2 = []
        model2 += [nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True), nn.BatchNorm2d(64),
                       nn.ReLU(True)]
        model2 += [nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True), nn.BatchNorm2d(64),
                       nn.ReLU(True)]
        model2 += [nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True), nn.BatchNorm2d(64),
                       nn.ReLU(True)]
        model2 += ResnetBlock()
        model2 += ResnetBlock()
        model2 += ResnetBlock()

        model2 += [nn.MultiHeadDotProductAttention(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        model2 += [nn.MultiHeadDotProductAttention(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        model2 += [nn.MultiHeadDotProductAttention(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        model2 += [nn.MultiHeadDotProductAttention(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]


        model2 += ResnetBlock()
        model2 += ResnetBlock()
        model2 += ResnetBlock()

        model2 += [nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True), nn.BatchNorm2d(64),nn.ReLU(True)]
        model2 += [nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True), nn.BatchNorm2d(64),nn.ReLU(True)]
        model2 += [nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=True)]

        self.model =  nn.Sequential(*model)
        self.model2 = nn.Sequential(*model2)


    def forward(self, bw_shadow, shadow ,mask):
        input1 = torch.cat((bw_shadow, mask),1)
        output1 = self.model(input1)
        input1 = torch.cat((output1, shadow, mask), 1)
        output2= self.model2(input2)
        return output1, output2

class ResnetBlock(nn.Module):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block()

    def build_conv_block(self):
        conv_block = []
        conv_block += [nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True), nn.BatchNorm2d(64), nn.ReLU(True)]
        conv_block += [nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True), nn.BatchNorm2d(64)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = [
            nn.Conv2d(input_nc, 64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64* 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64 * 2, 1, kernel_size=1, stride=1, padding=0, bias=True)]
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)



class SAM(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        if sa_type == 0:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1))
            self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
            self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.softmax = nn.Softmax(dim=-2)
        else:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_planes // share_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
            self.unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
            self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)

    def forward(self, x):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        if self.sa_type == 0:  # pairwise
            p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
            w = self.softmax(self.conv_w(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(x.shape[0], -1, 1, x.shape[2]*x.shape[3])
            x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, 1, x1.shape[-1])
            w = self.conv_w(torch.cat([x1, x2], 1)).view(x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
        x = self.aggregation(x3, w)
        return x



