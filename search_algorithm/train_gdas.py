import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as dset
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))

from search_space import utils
from search_algorithm.architect import Architect
from search_space.model_search import Network
from search_space.model_search_gumbel_softmax import Network_GumbelSoftmax
# don't remove this import
import search_space.genotypes

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--optimization', type=str, default='MiLeNAS', help='Optimization Methods: AOS; MiLeNAS')
parser.add_argument('--arch_search_method', type=str, default='DARTS',
                    help='Architecture Search Methods: DARTS; GDAS; DARTS_V2')
parser.add_argument('--tau_max', type=float, help='initial tau')
parser.add_argument('--tau_min', type=float, help='minimum tau')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

lambda_regularizer = 1

is_multi_gpu = False


def main():
    wandb.init(
        project="automl-gradient-based-nas",
        name="GDAS-" + "Opt: " + str(args.optimization) + "Search: " + str(args.arch_search_method),
        config=args,
        entity="automl"
    )

    wandb.config.update(args)  # adds all of the arguments as config variables

    global is_multi_gpu

    gpus = [int(i) for i in args.gpu.split(',')]
    logging.info('gpus = %s' % gpus)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # default: args.init_channels = 16, CIFAR_CLASSES = 10, args.layers = 8
    if args.arch_search_method == "DARTS":
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    elif args.arch_search_method == "GDAS":
        model = Network_GumbelSoftmax(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    else:
        raise Exception("search space does not exist!")

    if len(gpus) > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
        is_multi_gpu = True

    model.cuda()

    wandb.watch(model)

    arch_parameters = model.module.arch_parameters() if is_multi_gpu else model.arch_parameters()
    arch_params = list(map(id, arch_parameters))

    parameters = model.module.parameters() if is_multi_gpu else model.parameters()
    weight_params = filter(lambda p: id(p) not in arch_params,
                           parameters)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        weight_params,  # model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)

    # will cost time to download the data
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))  # split index

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=8)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=8)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, criterion, args)

    best_accuracy = 0

    table = wandb.Table(columns=["Epoch", "Searched Architecture"])

    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        genotype = model.module.genotype() if is_multi_gpu else model.genotype()
        logging.info('genotype = %s', genotype)
        wandb.log({"genotype": str(genotype)}, step=epoch)

        table.add_data(str(epoch), str(genotype))
        wandb.log({"Searched Architecture": table})

        print(F.softmax(model.module.alphas_normal if is_multi_gpu else model.alphas_normal, dim=-1))
        print(F.softmax(model.module.alphas_reduce if is_multi_gpu else model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(epoch, train_queue, valid_queue, model, architect, criterion, optimizer, lr)
        logging.info('train_acc %f', train_acc)
        wandb.log({"searching_train_acc": train_acc, "epoch": epoch})

        # validation
        with torch.no_grad():
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        wandb.log({"searching_valid_acc": valid_acc, "epoch": epoch})

        scheduler.step()

        if valid_acc > best_accuracy:
            wandb.run.summary["best_valid_accuracy"] = valid_acc
            best_accuracy = valid_acc

        # utils.save(model, os.path.join(args.save, 'weights.pt'))
        utils.save(model, os.path.join(wandb.run.dir, 'weights.pt'))


def train(epoch, train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    global is_multi_gpu

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        # logging.info("epoch %d, step %d START" % (epoch, step))
        model.train()
        n = input.size(0)

        model.set_tau(args.tau_max - epoch * 1.0 / args.epochs * (args.tau_max - args.tau_min))

        input = input.cuda()
        target = target.cuda()

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda()

        # Update architecture alpha by Adam-SGD
        # logging.info("step %d. update architecture by Adam. START" % step)
        #
        if args.optimization == "AOS":
            architect.step_AOS(input, target, input_search, target_search)
        else:
            architect.step_milenas(input, target, input_search, target_search, 1, 1)

        # logging.info("step %d. update architecture by Adam. FINISH" % step)
        # Update weights w by SGD, ignore the weights that gained during architecture training

        # logging.info("step %d. update weight by SGD. START" % step)
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        parameters = model.module.arch_parameters() if is_multi_gpu else model.arch_parameters()
        nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()

        # logging.info("step %d. update weight by SGD. FINISH\n" % step)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    global is_multi_gpu

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
