import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='PlaneMatch')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--save_snapshot', action='store_true', default=True)
    parser.add_argument('--save_snapshot_every', type=int, default=20)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--lr_decay_by', type=float, default=0.5)
    parser.add_argument('--lr_decay_every', type=float, default=250)
    parser.add_argument('--focal_loss_lambda', type=int, default=3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--train_csv_path', type=str, default='./data/triplet_train.csv')
    parser.add_argument('--test_csv_path', type=str, default='./data/triplet_test.csv')
    #parser.add_argument('--train_root_dir', type=str, default='/media/mahdi/4418B81419D11C10/media/private/dataset/PlaneMatch/training_triplets')
    parser.add_argument('--train_root_dir', type=str,
                        default='/media/mahdi/4418B81419D11C10/media/private/dataset/scannet/cop')
    parser.add_argument('--test_root_dir', type=str, default='/media/mahdi/4418B81419D11C10/media/private/dataset/scannet/cop-test')
    parser.add_argument('--save_path', type=str, default='./models')
    parser.add_argument('--feature_path', type=str, default='./feature_extraction')
    
    args = parser.parse_args()
    return args
