import argparse
import torch
from data.cifar100 import Cifar100

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar2 import Cifar2
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
import sys; sys.path.append("..")
from sam import SAM
import os
import torch.utils.data.distributed
import os
import torch
import torch.distributed as dist
import torch.utils.data.distributed
from torch.multiprocessing import Process
import numpy as np
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '18360'

def getbesthyperparameter(tensorlist,tensor):
    #share loss and tensor
    dist.all_gather(tensorlist,tensor)

def average_params(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        # dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=0)
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size

def main(rank):
    dist.init_process_group("gloo", rank=rank, world_size=args.train_workers)
    initialize(args, seed=42)
    torch.cuda.set_device(rank % 2)
    # 针对性修改batchsize
    batchsize = int(args.batch_size / (args.train_workers / 2))
    dataset = Cifar100(batchsize, args.threads)
    log = Log(log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=100).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank % 2])
    np.random.seed(rank)
    # mylr = args.learning_rate * (0.8 + 0.4 * np.random.rand())
    mylr = 0.1 * (rank + 1)
    print("rank",rank,"lr is",mylr)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=mylr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    #引入余弦表优化器leader
    if rank == 0:
        mylr = args.learning_rate
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=mylr,
                        momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=0, last_epoch=-1)

    #PANDA算法用参数
    tensor_list = [torch.zeros(2, dtype=torch.float).cuda() for _ in range(dist.get_world_size())]
    mytensor_list = [[0, 0] for _ in range(dist.get_world_size())]

    for epoch in range(args.epochs):
        dataset.train_sampler.set_epoch(epoch)
        model.train()
        log.train(len_dataset=len(dataset.train))
        epoch_accuracy = 0
        for batch in dataset.train:
            inputs, targets = (b.cuda() for b in batch)

            # first forward-backward step
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            smooth_crossentropy(model(inputs), targets).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), mylr)
                # log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                # scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.cuda() for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
            epoch_accuracy = log.epoch_state["accuracy"] / log.epoch_state["steps"]
        average_params(model)
        sharetensor = torch.tensor([epoch_accuracy, mylr]).cuda()
        getbesthyperparameter(tensor_list, sharetensor)
        for i in range(len(tensor_list)):
            mytensor_list[i] = tensor_list[i].tolist()
        print(mytensor_list)
        bestrank = mytensor_list.index(max(mytensor_list))
        if dist.get_rank() != bestrank:
            mylr = mytensor_list[bestrank][1] * (1 + 0.05*np.random.rand())
        if rank != 0:
            optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=mylr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            scheduler.step()
            mylr = optimizer.param_groups[0]['lr']



    log.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--train_workers", default=2, type=int, help="How many workers for training.")
    args = parser.parse_args()
    size = args.train_workers
    processes = []
    for rank in range(size):
        p = Process(target=main, args=(rank,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
