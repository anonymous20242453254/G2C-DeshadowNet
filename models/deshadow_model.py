import torch
from .base_model import BaseModel
from . import networks

class DeshadowModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'G_L2' 'D_real', 'D_fake']
        self.visual_names = ['deshadow_rgb','deshadow_bw']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.netG = networks.define_G(opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.shadow = input['shadow'].to(self.device)
        self.nonshadow = input['nonshadow'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.image_paths = input['shadow_paths']

    def forward(self):
        self.bw_shadow = 0.11 * self.shadow[:,0,:,:] + 0.31 * self.shadow[:,1,:,:] + 0.59 * self.shadow[:,2,:,:]
        self.bw_nonshadow = 0.11 * self.nonshadow[:,0,:,:] + 0.31 * self.nonshadow[:,1,:,:] + 0.59 * self.nonshadow[:,2,:,:]
        self.bw_shadow = self.bw_shadow.view(1,1,self.shadow.shape[2],self.shadow.shape[3])
        self.deshadow_bw, self.deshadow_rgb = self.netG(self.bw_shadow, self.shadow, self.mask)

    def backward_D(self):
        pred_fake = self.netD(self.bw_shadow.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        pred_real = self.netD(self.bw_nonshadow)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        pred_fake = self.netD(self.bw_shadow)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.deshadow_bw, self.bw_nonshadow) * 100
        self.loss_G_L2 = self.criterionL1(self.deshadow_rgb, self.nonshadow) * 100

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_L2
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)

        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
