#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import imp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, femnist_star, cifar_noniid_2, cifar_100_iid, \
    cifar_100_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNMNIST, CNN2Cifar, CNNCifar100Std5
from models.Fed import FedAvg
from models.test import test_img, save_result_img, xferable_to_state_dict, state_dict_to_xferable
import json
import socket
import time


if __name__ == '__main__':


    def get_devices_info( filename ):
        with open(filename) as f:
            string = f.readline()
        json_obj = json.loads(string)
        return json_obj



    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:  # non-IID
            dict_users = cifar_noniid_2(dataset_train, args.num_users)
        # else:
        #     exit('Error: only consider IID setting in CIFAR10')

    elif args.dataset == 'cifar100':
        _CIFAR_TRAIN_TRANSFORMS = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ]
        dataset_train = datasets.CIFAR100(
            '../data/cifar100', train=True, download=True,
            transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS))
        _CIFAR_TEST_TRANSFORMS = [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ]
        dataset_test = datasets.CIFAR100(
            '../data/cifar100', train=False,
            transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS))
        if args.iid:  # IID
            dict_users = cifar_100_iid(dataset_train, args.num_users)
        else:  # non-IID
            dict_users = cifar_100_noniid(dataset_train, args.num_users)

    elif args.dataset == 'emnist':
        _MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        dataset_train = datasets.EMNIST(
            '../data/emnist', train=True, download=True,
            transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS),
            split='letters'
        )
        dataset_test = datasets.EMNIST(
            '../data/emnist', train=False, download=True,
            transform=transforms.Compose(_MNIST_TEST_TRANSFORMS),
            split='letters'
        )

        dict_users = femnist_star(dataset_train, args.num_users)

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar100':
        net_glob = CNNCifar100Std5(args=args).to(args.device)
    # elif args.model == 'cnn' and args.dataset == 'emnist':
    #     net_glob = CNNEmnistStd5(args=args).to(args.device)
    elif args.model == 'cnn2' and args.dataset == 'cifar':
        net_glob = CNN2Cifar(args=args).to(args.device)
    elif args.model == 'cnn2' and args.dataset == 'mnist':
        net_glob = CNNMNIST(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    devices = get_devices_info("FL_nodes.json")

    m = max(int(args.frac * args.num_users), 1)

    idxs_users = np.random.choice(range(args.num_users),
                                       m,
                                       replace=False)
    # Socks list
    socks = [None] * len(idxs_users)

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    acc_tr_list, loss_tr_list, acc_test_list, loss_test_list = [], [], [], []
    # MSGLEN = 484449

    if args.all_clients: 
        print("Aggregation over all clients")
        # w_locals = [w_glob for i in range(args.num_users)]
        w_locals = []
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for i, idx in enumerate(idxs_users):

            xfer_dict = state_dict_to_xferable(net_glob.state_dict(),
                                               lr=args.lr,
                                               ep=iter)

            if idx in args.remote_index:
                node_name = "node%d" % idx
                HOST = devices[node_name]['ip']
                PORT = devices[node_name]['port']
                print(HOST)
                print(PORT)
                from client import MSGLEN
                w_bytes = bytes(xfer_dict + "\n" + " " * (MSGLEN - len(xfer_dict) - 1), "utf-8")

                MSGLEN = len(w_bytes)
                print(len(w_bytes))
                # if len(w_bytes) != MSGLEN:
                #     raise Exception("w_bytes should be equal to MSGLEN(%d)" % MSGLEN)

                # Create a socket (SOCK_STREAM means a TCP socket)
                socks[i] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                socks[i].connect((HOST, PORT))
                socks[i].sendall(w_bytes)

                # Receive the updated model from all remote workers
                elapsed = []
                for i, idx in enumerate(idxs_users):
                    if idx in args.remote_index:
                        node_name = "node%d" % idx
                        HOST = devices[node_name]['ip']
                        PORT = devices[node_name]['port']

                        # Receive updated model from worker nodes
                        chunks = []
                        bytes_recd = 0
                        while bytes_recd < MSGLEN:
                            chunk = socks[i].recv(min(MSGLEN - bytes_recd, 8096))
                            if chunk == b'':
                                raise RuntimeError("socket connection broken")
                            chunks.append(chunk)
                            bytes_recd = bytes_recd + len(chunk)

                        # Decode the received bytes into formatted string
                        data = b''.join(chunks)
                        w_str = data.decode("utf-8")

                        # Decode the formatted string to the model
                        w, lr, ep = xferable_to_state_dict(w_str)
                        w_locals.append(copy.deepcopy(w))
                        print(type(w))

                # Ensure receiving all the updated weight
                # if len(w_locals) != len(idxs_users):
                #     err_msg = "w_locals only has %d weight update (Ideally should be %d" % (len(self.w_locals), len(self.idxs_users))
                #     raise ValueError(err_msg)
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        # loss_avg = sum(loss_locals) / len(loss_locals)
        # print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        # loss_train.append(loss_avg)

        # test for per epoch
        net_glob.eval()
        acc_tr, loss_tr = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)

        # completed list for visualize
        acc_tr_list.append(acc_tr)
        loss_tr_list.append(loss_tr)
        acc_test_list.append(acc_test)
        loss_test_list.append(loss_test)
        
    # save result
    save_result_img(args, acc_tr_list, name="train accurcy")
    save_result_img(args, loss_tr_list, name="train loss")
    save_result_img(args, acc_test_list, name="test accurcy")
    save_result_img(args, loss_test_list, name="test loss")

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    # print("Testing accuracy: {:.2f}".format(acc_test))

