import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Classification of SalUn Experiments")

    ##################################### Dataset #################################################
    parser.add_argument(
        "--data", type=str, default="../data", help="location of the data corpus"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument(
        "--input_size", type=int, default=32, help="size of input images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./tiny-imagenet-200",
        help="dir to tiny-imagenet",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)

    ##################################### Architecture ############################################
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="model architecture"
    )
    parser.add_argument(
        "--imagenet_arch",
        action="store_true",
        help="architecture for imagenet size samples",
    )
    parser.add_argument(
        "--train_y_file",
        type=str,
        default="./labels/train_ys.pth",
        help="labels for training files",
    )
    parser.add_argument(
        "--val_y_file",
        type=str,
        default="./labels/val_ys.pth",
        help="labels for validation files",
    )

    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument(
        "--train_seed",
        default=1,
        type=int,
        help="seed for training (default value same as args.seed)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers in dataloader"
    )
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument(
        "--save_dir",
        help="The directory used to save the trained models",
        default=None,
        type=str,
    )
    parser.add_argument("--model_path", type=str, default=None, help="the path of original model")

    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--epochs", default=182, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
    parser.add_argument(
        "--no-aug",
        action="store_true",
        default=False,
        help="No augmentation in training dataset (transformation).",
    )
    parser.add_argument("--no-l1-epochs", default=0, type=int, help="non l1 epochs")

    ##################################### Pruning setting #################################################
    parser.add_argument("--prune", type=str, default="omp", help="method to prune")
    parser.add_argument(
        "--pruning_times",
        default=1,
        type=int,
        help="overall times of pruning (only works for IMP)",
    )
    parser.add_argument(
        "--rate", default=0.95, type=float, help="pruning rate"
    )  # pruning rate is always 20%
    parser.add_argument(
        "--prune_type",
        default="rewind_lt",
        type=str,
        help="IMP type (lt, pt or rewind_lt)",
    )
    parser.add_argument(
        "--random_prune", action="store_true", help="whether using random prune"
    )
    parser.add_argument("--rewind_epoch", default=0, type=int, help="rewind checkpoint")
    parser.add_argument(
        "--rewind_pth", default=None, type=str, help="rewind checkpoint to load"
    )

    ##################################### Unlearn setting #################################################
    parser.add_argument(
        "--unlearn", type=str, default="retrain", help="method to unlearn"
    )
    parser.add_argument(
        "--unlearn_lr", default=0.01, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--unlearn_epochs",
        default=10,
        type=int,
        help="number of total epochs for unlearn to run",
    )

    parser.add_argument(
        "--num_indexes_to_replace",
        type=int,
        default=None,
        help="Number of data to forget",
    )
    parser.add_argument(
        "--class_to_replace", type=int, default=-1, help="Specific class to forget"
    )

    parser.add_argument(
        "--indexes_to_replace",
        type=list,
        default=None,
        help="Specific index data to forget",
    )
    parser.add_argument("--alpha", default=0.2, type=float, help="unlearn noise & IU &")
    parser.add_argument("--mask_path", default=None, type=str, help="the path of saliency map")
    parser.add_argument("--type", type=str, default="full_class", help="Specific unlearn type for cifar100 class_wise")
    parser.add_argument("--retrain_model_path", default=None,type=str,help="retrained_model_path is none")
    parser.add_argument("--top_data", default=0.1, type=float, help = "miss the top percent of chosen new forget data")
    parser.add_argument('--forget_pid', type=int, default=None, help='need the reid to be forgotten')
    
    ##################################### SSD parameters #################################################
    parser.add_argument('--ssd_lower_bound',       type=float, default=1.0)
    parser.add_argument('--ssd_exponent',          type=float, default=1.0)
    parser.add_argument('--ssd_magnitude_diff',    type=float, default=0.0)
    parser.add_argument('--ssd_forget_threshold',  type=float, default=1.0)
    parser.add_argument('--ssd_dampening_constant',type=float, default=1.0)
    parser.add_argument('--ssd_selection_weighting',type=float, default=10.0)
    parser.add_argument("--ssd_max_layer", type=float, default=-1)
    parser.add_argument("--ssd_min_layer", type=float, default=-1)

    ##################################### SCAR parameters #################################################
    parser.add_argument("--surrogate_dataset", type=str, default='subset_Imagenet')
    parser.add_argument("--surrogate_quantity", type=int, default=-1,help='-1 for all data,1 for 1k data,2 for 2k data,..., 10 for 10k data')
    parser.add_argument("--bsize", type=int, default=1024)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--lambda_1", type=float, default=1)
    parser.add_argument("--lambda_2", type=float, default=5)
    parser.add_argument("--scheduler", type=int, nargs='+', default=[25,40])
    parser.add_argument("--delta", type=float, default=.5)
    parser.add_argument("--gamma1", type=float, default=3)
    parser.add_argument("--gamma2", type=float, default=3)
    parser.add_argument("--scar_epochs", default=1, type=int, help="number of total epochs for unlearn to run")
    parser.add_argument("--surrogate_dataset_path", type=str, default=None, help="need the path of surrogate dataset")
    
    ##################################### NPO parameters #################################################
    parser.add_argument("--beta", type=float, default=0, help="need value of beta in algorithm")
    parser.add_argument("--unlearned_model", default=None, type=str, help="the path of unlearned model in person re-id task")

    
    
    return parser.parse_args()
