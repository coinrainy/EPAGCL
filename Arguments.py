import argparse
import os

from datetime import datetime


def get_args():
    parser = argparse.ArgumentParser()

    # Training Settings
    # ----------------------------------------------------------------
    parser.add_argument('--cuda_num', type=int, default=0,
                        help='CUDA to train on.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--repeat', type=int, default=5,
                        help='repeating times')
    parser.add_argument('--not_add_edge', action="store_true")
    parser.add_argument('--not_drop_edge', action="store_true")
    parser.add_argument('--not_drop_feature_random', action="store_true")
    parser.add_argument('--add_single', action='store_true')
    parser.add_argument('--add_edge_random', action="store_true")
    parser.add_argument('--batch_compute', action="store_true")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fixed_split', action="store_true", default=True)

    # optimizer
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # log and saving
    parser.add_argument('--path', type=str, default='./trained_models/',
                        help='The path of outputs')
    parser.add_argument('--loss_log', type=int, default=1,
                        help='Frequency to log loss.')
    parser.add_argument('--eval', type=int, default=50,
                        help='Frequency to eval model.')
    parser.add_argument('--emit_json', action="store_true",
                        help='Print one AUTORESEARCH_RESULT JSON line at the end.')

    # Dataset Settings
    # ----------------------------------------------------------------
    parser.add_argument('--dataset', type=str, default='Cora')
    default_dataset_path = '/root/datasets/' if os.path.exists('/root/datasets') else '../datasets/'
    parser.add_argument('--dataset_path', type=str, default=default_dataset_path)

    # augmentation
    parser.add_argument('--edge_add_rate', type=float, default=0.1)
    parser.add_argument('--edge_drop_rate_1', type=float, default=0.2)
    parser.add_argument('--edge_drop_rate_2', type=float, default=0.3)
    parser.add_argument('--feat_drop_rate_1', type=float, default=0.1)
    parser.add_argument('--feat_drop_rate_2', type=float, default=0.1)

    # Networks Settings
    # ----------------------------------------------------------------
    parser.add_argument('--feat_dim', type=int, default=256)
    parser.add_argument('--proj_hidden_dim', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.3)

    # Evaluate Settings
    # ----------------------------------------------------------------
    parser.add_argument('--eval_epoch', type=int, default=3000)
    parser.add_argument('--eval_optim', type=str, default='Adam')
    parser.add_argument('--eval_lr', type=float, default=0.01)
    parser.add_argument('--eval_weight_decay', type=float, default=0.0)

    args = parser.parse_args()
    args.device = 'cuda:' + str(args.cuda_num)
    args.output_path = args.path + args.dataset + '/in-progress_' + datetime.now().strftime('%m%d_%H:%M:%S')
    args.dataset_path += args.dataset
    args.save_model = max(1, args.epoch // 10)
    args.add_edge = not args.not_add_edge
    args.drop_edge = not args.not_drop_edge
    args.drop_feature_random = not args.not_drop_feature_random

    for i in range(args.repeat):
        path = args.output_path + '/' + str(i + 1)
        if not os.path.exists(path):
            os.makedirs(path)

    with open(args.output_path + '/args.txt', "a") as f:
        for k, v in args.__dict__.items():
            print(f"{k}: {v}", file=f)

    with open(args.output_path + '/result.txt', "a") as f:
        print(f"dataset: {args.dataset}", file=f)
        print(f"drop edge: {args.drop_edge}", file=f)
        print(f"add edge: {args.add_edge}", file=f)
        if args.add_edge:
            print(f"add edge for single graph: {args.add_single}", file=f)
        print(f"drop feature randomly: {args.drop_feature_random}", file=f)
    return args
