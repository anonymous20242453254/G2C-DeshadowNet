from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400)
        parser.add_argument('--display_ncols', type=int, default=4)
        parser.add_argument('--display_id', type=int, default=1)
        parser.add_argument('--display_server', type=str, default="http://localhost")
        parser.add_argument('--display_env', type=str, default='main')
        parser.add_argument('--display_port', type=int, default=8097)
        parser.add_argument('--update_html_freq', type=int, default=1000)
        parser.add_argument('--print_freq', type=int, default=100)
        parser.add_argument('--no_html', action='store_true')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000)
        parser.add_argument('--save_epoch_freq', type=int, default=5)
        parser.add_argument('--save_by_iter', action='store_true')
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--epoch_count', type=int, default=1)
        parser.add_argument('--phase', type=str, default='train')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100)
        parser.add_argument('--n_epochs_decay', type=int, default=100)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--lr_decay_iters', type=int, default=100)
        self.isTrain = True
        return parser
