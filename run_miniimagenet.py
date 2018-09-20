"""
Train a model on miniImageNet.
"""

import random
import os
import tensorflow as tf

from supervised_mtl.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from supervised_mtl.eval import evaluate
from supervised_mtl.models import MiniImageNetModel,MiniImageNetMetaTransferModel
from supervised_mtl.miniimagenet import read_dataset
from supervised_mtl.train import train

DATA_DIR = '/home/erfan/miniimagenet'

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    random.seed(args.seed)

    train_set, val_set, test_set = read_dataset(DATA_DIR)
    if args.metatransfer:
        model = MiniImageNetMetaTransferModel(args.classes, **model_kwargs(args))
    else:
        model = MiniImageNetModel(args.classes, **model_kwargs(args))
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if not args.pretrained:
            print('Training...')
            train(sess, model, train_set, test_set, args.checkpoint, **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
#        print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
#        print('Validation accuracy: ' + str(evaluate(sess, model, val_set, **eval_kwargs)))
        print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))

if __name__ == '__main__':
    main()
