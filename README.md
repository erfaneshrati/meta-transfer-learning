# Meta-transfer learning

Meta-transfer learning (MTL) algorithm is an approach for training models which combines meta-learning and transfer learning gradient updates to acheive better accuracies in both few-shot and many-shot settings. It works by adding a fixed discriminator layer over the whole dataset to the meta-learner(Reptile or FOMAML). This repository is developed on OpenAI Reptile repository.

# Getting the data

The [fetch_data.sh](fetch_data.sh) script creates a `data/` directory and downloads miniImageNet into it.

```
$ ./fetch_data.sh
Fetching Mini-ImageNet train set ...
Fetching wnid: n01532829
Fetching wnid: n01558993
Fetching wnid: n01704323
Fetching wnid: n01749939
...
```

# Training

Here are some example commands:

```shell
# 5-ways MTL Reptile.
python -u run_miniimagenet.py --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 100000 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --classes 5 --checkpoint model_checkpoints/reptile_metatransfer_5ways --gpu 0 --metatransfer

# 20-ways MTL FOML.
python -u run_miniimagenet.py --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 100000 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --classes 20 --checkpoint model_checkpoints/foml_metatransfer_20ways --gpu 0 --metatransfer --foml

# 5-ways Reptile.
python -u run_miniimagenet.py --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 100000 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --classes 5 --checkpoint model_checkpoints/reptile_5ways --gpu 0

# 20-ways FOML.
python -u run_miniimagenet.py --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 100000 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --classes 20 --checkpoint model_checkpoints/foml_20ways --gpu 0 --foml
```

For more information about the training args we refer to Reptile repo.

# Testing

By adding --pretrained argument to the runner we can test the pre-trained model for unseen tasks:

```shell
# 5-ways 1-shot MTL Reptile.
python -u run_miniimagenet.py --shots 1 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --classes 5 --checkpoint model_checkpoints/reptile_metatransfer_5ways --transductive --gpu 0 --metatransfer --pretrained

# 20-ways 1-shot MTL FOML.
python -u run_miniimagenet.py --shots 1 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --classes 20 --checkpoint model_checkpoints/reptile_metatransfer_20ways --transductive --gpu 0 --metatransfer --foml --pretrained

# 5-ways 1-shot Reptile.
python -u run_miniimagenet.py --shots 1 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --classes 5 --checkpoint model_checkpoints/reptile_5ways --transductive --gpu 0 --pretrained

# 20-ways 1-shot FOML.
python -u run_miniimagenet.py --shots 1 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --classes 20 --checkpoint model_checkpoints/reptile_20ways --transductive --gpu 0 --foml --pretrained
```
