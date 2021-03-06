#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_result_img(args, in_list, name):
        plt.figure()
        plt.plot(range(len(in_list)), in_list)
        plt.ylabel(name)
        plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, name))


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0

    correct_list = []
    loss_list = []

    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        
        # store correct list for report
        correct_list.append(100.00 * correct / len(data_loader.dataset))
        # loss_list.append(test_loss)


    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    # save_result_img(args, correct_list, name="test accurcy")
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def state_dict_to_xferable( state_dict, lr=None, ep=None ):
    import json
    xfer_dict = {}
    if lr is not None:
        xfer_dict['lr'] = lr
    if ep is not None:
        xfer_dict['ep'] = ep
    for k in state_dict:
        xfer_dict[k] = state_dict[k].cpu().tolist()
    xfer_string = json.dumps(xfer_dict)
    return xfer_string


def xferable_to_state_dict(xfer_string):
    import json
    import collections
    json_decode_dict = json.loads(xfer_string)
    recovered_dict = collections.OrderedDict()
    for k in json_decode_dict:
        if k != 'lr' and k != 'ep':
            recovered_dict[k] = torch.Tensor(json_decode_dict[k])
    print("recovered_dict: ", type(recovered_dict))
    return recovered_dict, json_decode_dict['lr'], json_decode_dict['ep']

