#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from models.Nets import CNNMNIST, CNNMnist, CNN2Cifar, CNNCifar, CNNCifar100Std5, MLP
import json
import time
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import socketserver
from utils.sampling import cifar_100_iid, cifar_100_noniid, cifar_iid, cifar_noniid_2, mnist_iid, mnist_noniid
from utils.options import args_parser
from models.test import xferable_to_state_dict, state_dict_to_xferable
import copy


MSGLEN = 20000000

# def state_dict_to_xferable(state_dict, lr=None, ep=None):
#     import json
#     xfer_dict = {}
#     if lr is not None:
#         xfer_dict['lr'] = lr
#     if ep is not None:
#         xfer_dict['ep'] = ep
#     for k in state_dict:
#         xfer_dict[k] = state_dict[k].cpu().tolist()
#     xfer_string = json.dumps(xfer_dict)
#     return xfer_string


# def xferable_to_state_dict(xfer_string):
#     import json
#     import collections
#     json_decode_dict = json.loads(xfer_string)
#     recovered_dict = collections.OrderedDict()
#     for k in json_decode_dict:
#         if k != 'lr' and k != 'ep':
#             recovered_dict[k] = torch.Tensor(json_decode_dict[k])
#     return recovered_dict, json_decode_dict['lr'], json_decode_dict['ep']


def get_dataset_from_name(dataset_name, iid, num_users):
    # load dataset and split users
    if dataset_name == 'mnist':

        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

    # elif dataset_name == 'emnist':

    #     dataset_train = torch.load('./datasets/emnist/processed/training_byclass.pt')
    #     dataset_test = torch.load('./datasets/emnist/processed/test_byclass.pt')

    else:
        raise ValueError("Invalid dataset name")

    return dataset_train, dataset_test


def get_model_from_args(args):
    # build model
    if args.dataset == 'mnist':
        if args.model == 'CNN2':
            net_glob = CNNMnist(args)
    # elif args.dataset == 'emnist':
    #     if args.model == 'CNNStd5':
    #         net_glob = CNNEmnistStd5(args)
    else:
        exit('Error: unrecognized model')
    return net_glob


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # image, label = self.dataset[self.idxs[item]]
        image, label = self.dataset[0][self.idxs[item]], self.dataset[1][self.idxs[item]]
        m = nn.ConstantPad2d(2, 0)
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(image.ndimension()))

            elif image.ndimension() == 2:
                # if 2D image, add channel dimension (CHW)
                pic = image.unsqueeze(0)
        if isinstance(pic, torch.Tensor):
            npimg = np.transpose(pic.numpy(), (1, 2, 0))
        npimg = npimg[:, :, 0]
        pic = Image.fromarray(npimg, mode='L')
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[1], pic.size[0], 1)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            image = img.float().div(255)
        else:
            image = img
        image = m(image)
        image = image.clone()
        dtype = image.dtype
        mean = torch.Tensor((0.1307,))
        std = torch.Tensor((0.3081,))
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image, label


class Dict_to_namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)



class MyTCPHandler(socketserver.BaseRequestHandler):
    
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

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape


    def handle(self):
        # MSGLEN = 15000000
        # MSGLEN = 9838544
        # MSGLEN = 160000

        self.dtype_dict = {}
        self.shape_dict = {}

        # self.request is the TCP socket connected to the client
        chunks = []
        bytes_recd = 0
        start_time = time.time()
        while bytes_recd <= MSGLEN:
            print("vared mishavad: ", bytes_recd)
            chunk = self.request.recv(min(MSGLEN - bytes_recd, 8096))
            if not chunk:
                break
            if bytes_recd == 0:
                time_model_recv_start = time.time()

            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)

        # time_model_xmit = time.time() - time_model_recv_start
        # print("Model transmission time = %.3f(s)" % time_model_xmit)

        # Decode received bytes to formatted string
        print(len(chunks))
        data = b''.join(chunks)
        self.data = data.decode("utf-8")
        print(len(self.data))
        
        print("{} wrote:".format(self.client_address[0]))

        # build model
        if self.args.model == 'cnn' and self.args.dataset == 'cifar':
            net_glob = CNNCifar(args=self.args).to(self.args.device)
        elif self.args.model == 'cnn' and self.args.dataset == 'mnist':
            net_glob = CNNMnist(args=self.args).to(self.args.device)
        elif self.args.model == 'cnn' and self.args.dataset == 'cifar100':
            net_glob = CNNCifar100Std5(args=self.args).to(self.args.device)
        # elif args.model == 'cnn' and args.dataset == 'emnist':
        #     net_glob = CNNEmnistStd5(args=args).to(args.device)
        elif self.args.model == 'cnn2' and self.args.dataset == 'cifar':
            net_glob = CNN2Cifar(args=self.args).to(self.args.device)
        elif self.args.model == 'cnn2' and self.args.dataset == 'mnist':
            net_glob = CNNMNIST(args=self.args).to(self.args.device)
        elif self.args.model == 'mlp':
            len_in = 1
            for x in self.img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=self.args.num_classes).to(self.args.device)
        else:
            exit('Error: unrecognized model')
        
        param, lr, ep = xferable_to_state_dict(self.data)
        net_glob.load_state_dict(param)
        from models.Update import LocalUpdate

        for name, param in net_glob.named_parameters():
            self.dtype_dict[name] = param.detach().cpu().numpy().dtype
            self.shape_dict[name] = param.detach().cpu().numpy().shape

        m = max(int(self.args.frac * self.args.num_users), 1)
        idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
        for i, idx in enumerate(idxs_users):
            local = LocalUpdate(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[idx])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(self.args.device))
        # if self.args.all_clients:
        #     w_locals[idx] = copy.deepcopy(w)
        # else:
        #     w_locals.append(copy.deepcopy(w))
        # loss_locals.append(copy.deepcopy(loss))

        new_model = state_dict_to_xferable(net_glob.state_dict(), lr=self.args.lr, ep=self.args.local_ep)
        # Perform a local epoch
        # w_str = FL_node.train(self.data)
        w_str = new_model
        # Encode to utf-8 and send back new model
        w_bytes = bytes(w_str + "\n", "utf-8")
        w_bytes += bytes(" ", "utf-8") * (MSGLEN - len(w_bytes))
        self.request.sendall(w_bytes)


def get_idxs_from_json(filename, usr_index):
    import json
    with open(filename) as f:
        dict_json_users = json.loads(f.readline())
    idxs = np.array(dict_json_users[str(usr_index)])
    return idxs


def main():
    import argparse
    from utils.options import args_parser

    args = args_parser()
    if args.usr_index is None:
        raise ValueError("args.usr_index is required")

    # Create the FL node and make it global, it will be called for training
    global FL_node
    idxs = get_idxs_from_json('dict_users.json', args.usr_index)

    # FL_node = LocalUpdate(args)
    # FL_node.set_dataset(args.dataset, idxs)
    # FL_node.set_model()

    # Create the server, binding to localhost on port 9999
    if args.port is None:
        raise ValueError("Need to specify port for compute node")
    HOST, PORT = args.host, args.port
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()


if __name__ == '__main__':
    main()
