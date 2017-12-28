from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.load_network(self.netG, 'G', opt.which_epoch)

        print('------------ Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        self.input_A.resize_as_(input_A).copy_(input_A)
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('real_A', real_A),
                            ('fake_B', fake_B)
                            ])

    # get image paths
    def get_image_paths(self):
        return self.image_paths