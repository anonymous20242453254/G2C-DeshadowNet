
class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass

def get_transform(opt):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)

def get_transform_mask(opt):
    transform_list = []
    transform_list += [transforms.ToTensor()]

    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def dataread(filelist):
    with open('filelist', encoding='utf8') as f:
        list1 = f.read().split('\n')
    return list1