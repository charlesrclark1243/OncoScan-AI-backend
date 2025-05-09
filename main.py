# Author: Charles R. Clark
# CS 6440 Spring 2024

from typing import Tuple, List
import argparse
import yaml
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassRecall

import data.data_utils as data_utils
from data.dataset_mri import Mri
from models import BaselineModel, FinalModel

def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn: any, optimizer: optim.Optimizer, epoch: int, nclasses=3) -> None:
    nbatches = len(dataloader)
    
    # avg_loss = Average()
    # avg_prec = Average()

    loss_dict = {
        'sum': 0.0,
        'nrecords': 0.0,
        'avg': 0.0
    }
    recall_dict = {
        'sum': 0.0,
        'nrecords': 0.0,
        'avg': 0.0
    }

    overall_metric = MulticlassRecall(num_classes=nclasses, average='micro', top_k=1).to(torch.device('cuda', 0))

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # ---------------------------------------------------------------
        # CUDA logic...
        # ---------------------------------------------------------------
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        # ---------------------------------------------------------------
        # forward pass...
        # ---------------------------------------------------------------
        out = model(X)
        loss = loss_fn(out, y)
        recall = overall_metric(out, y)

        # ---------------------------------------------------------------
        # update average dicts...
        # ---------------------------------------------------------------

        loss_dict['sum'] += loss
        loss_dict['nrecords'] += 1.0
        loss_dict['avg'] = loss_dict['sum'] / loss_dict['nrecords']

        recall_dict['sum'] += recall
        recall_dict['nrecords'] += 1.0
        recall_dict['avg'] = recall_dict['sum'] / recall_dict['nrecords']
        
        # ---------------------------------------------------------------
        # backward pass...
        # ---------------------------------------------------------------
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # ---------------------------------------------------------------
        # print statements...
        # ---------------------------------------------------------------
        if batch % 50 == 0:
            print(f'Epoch: {epoch} [{batch}/{nbatches}]\t' + \
                  f'Loss: {loss:.4f} [{loss_dict["avg"]:.4f}]\t' + \
                  f'Recall @1: {recall:.4f} [{recall_dict["avg"]:.4f}]')
            
    print() # skip a line in print pattern

def validate_loop(dataloader: DataLoader, model: nn.Module, loss_fn: any, epoch: int, nclasses=3) -> Tuple[float, List[float]]:
    nbatches = len(dataloader)

    # avg_overall_loss = Average()
    # avg_overall_prec = Average()
    # avg_class_precs = [Average()] * nclasses

    loss_dict = {
        'sum': 0.0,
        'nrecords': 0.0,
        'avg': 0.0
    }
    recall_dict = {
        'sum': 0.0,
        'nrecords': 0.0,
        'avg': 0.0
    }
    class_recalls = [copy.deepcopy(recall_dict) for _ in range(nclasses)]

    overall_metric = MulticlassRecall(num_classes=nclasses, average='micro', top_k=1).to(torch.device('cuda', 0))
    perclass_metric = MulticlassRecall(num_classes=nclasses, average=None, top_k=1).to(torch.device('cuda', 0))

    model.eval()
    with torch.no_grad():
        batch = 0
        for X, y in dataloader:
            # ---------------------------------------------------------------
            # CUDA logic...
            # ---------------------------------------------------------------
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            # ---------------------------------------------------------------
            # forward pass...
            # ---------------------------------------------------------------
            out = model(X)
            loss = loss_fn(out, y)
            overall_recall = overall_metric(out, y)
            perclass_recalls = perclass_metric(out, y)

            # avg_overall_loss.add_to_sum(loss)
            # avg_overall_prec.add_to_sum(prec)

            # ---------------------------------------------------------------
            # update average dicts...
            # ---------------------------------------------------------------

            loss_dict['sum'] += loss
            loss_dict['nrecords'] += 1.0
            loss_dict['avg'] = loss_dict['sum'] / loss_dict['nrecords']

            recall_dict['sum'] += overall_recall
            recall_dict['nrecords'] += 1.0
            recall_dict['avg'] = recall_dict['sum'] / recall_dict['nrecords']

            for class_idx in range(nclasses):
                class_recall = perclass_recalls[class_idx]
                
                class_recalls[class_idx]['sum'] += class_recall
                class_recalls[class_idx]['nrecords'] += 1.0
                class_recalls[class_idx]['avg'] = class_recalls[class_idx]['sum'] / class_recalls[class_idx]['nrecords']

            # for class_idx in range(nclasses):
            #     class_prec = precision_at_k(out, y, class_idx=class_idx)

            #     if not (class_prec < 0.0):
            #         avg_class_precs[class_idx].add_to_sum(class_prec)

            #     print(f'{class_idx} ---> {avg_class_precs[class_idx].sum}, {avg_class_precs[class_idx].nrecords}: {avg_class_precs[class_idx].avg} ({class_prec})')

            # ---------------------------------------------------------------
            # print statements...
            # ---------------------------------------------------------------
            if batch % 50 == 0:
                print(f'Epoch: {epoch} [{batch}/{nbatches}]\t' + \
                      f'Loss: {loss:.4f} [{loss_dict["avg"]:.4f}]\t' + \
                      f'Recall @1: {overall_recall:.4f} [{recall_dict["avg"]:.4f}]')
                
            batch += 1

    # ---------------------------------------------------------------
    # final print statements...
    # ---------------------------------------------------------------
    print()

    for class_idx in range(nclasses):
        print(f'Recall @1 of class {class_idx}: {class_recalls[class_idx]["avg"]:.4f}')

    print(f'\nOverall Recall @1: {recall_dict["avg"]:.4f}')
    print('---------------------------------------------------------------')

    return recall_dict['avg'], [class_recalls[class_idx]['avg'] for class_idx in range(nclasses)]

parser = argparse.ArgumentParser(description='Argument parser for passing a configuration .yaml file')
parser.add_argument('-c', '--config')

def main():
    global args
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    for outer_key in config:
        for inner_key in config[outer_key]:
            setattr(args, inner_key, config[outer_key][inner_key])

    # ---------------------------------------------------------------
    # define data transforms...
    # ---------------------------------------------------------------
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.img_size, args.img_size)),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Normalize((0.5, ), (0.5))
    ])

    # ---------------------------------------------------------------
    # load data...
    # ---------------------------------------------------------------

    normal = data_utils.import_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'Normal'), label=data_utils.MAPPING['normal'])
    glioma = data_utils.import_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'Tumor', 'glioma_tumor'), label=data_utils.MAPPING['glioma'])
    meningioma = data_utils.import_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'Tumor', 'meningioma_tumor'), label=data_utils.MAPPING['meningioma'])

    df = data_utils.combine_and_shuffle_data([normal, glioma, meningioma])
    train_df, test_df = data_utils.split_data(df=df)
    
    train_mri = Mri(df=train_df, transform=transforms)
    test_mri = Mri(df=test_df, transform=transforms)

    train_dataloader = DataLoader(train_mri, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_mri, batch_size=args.batch_size, shuffle=True)

    # ---------------------------------------------------------------
    # initialize model...
    # ---------------------------------------------------------------

    if args.model == 'BaselineModel':
        model = BaselineModel(init_img_size=args.img_size)
    else:
        model = FinalModel(init_img_size=args.img_size)
    
    if torch.cuda.is_available():
        model = model.cuda()

    # ---------------------------------------------------------------
    # initiailize loss and optimizer...
    # ---------------------------------------------------------------

    loss_fn = nn.CrossEntropyLoss()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     args.learning_rate,
                                     betas=tuple(args.betas),
                                     eps=args.epsilon,
                                     weight_decay=args.weight_decay)
    elif args.optimzier == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)
    else:
        print(f'\nInvalid Optimizer Argument: {args.optimizer}\n')
        exit(-1)

    # ---------------------------------------------------------------
    # main optimizer loop...
    # ---------------------------------------------------------------

    best_avg_overall_recall = 0.0
    best_avg_class_recalls = None
    best_model = None
    for epoch in range(args.epochs):    
        # ---------------------------------------------------------------
        # train...
        # ---------------------------------------------------------------
        train_loop(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch)

        # ---------------------------------------------------------------
        # validate...
        # ---------------------------------------------------------------
        avg_overall_recall, avg_class_recalls = validate_loop(dataloader=test_dataloader, model=model, loss_fn=loss_fn, epoch=epoch)

        # ---------------------------------------------------------------
        # save best model...
        # ---------------------------------------------------------------
        if avg_overall_recall > best_avg_overall_recall:
            best_avg_overall_recall = avg_overall_recall
            best_avg_class_recalls = avg_class_recalls[:]
            best_model = copy.deepcopy(model)

    # ---------------------------------------------------------------
    # print statements...
    # ---------------------------------------------------------------
    print('\n===============================================================')
    print(f'Best Recall @1: {best_avg_overall_recall:.4f}')
    print('===============================================================\n')
    
    for class_idx in range(len(best_avg_class_recalls)):
        print(f'Recall @1 of class {class_idx}: {best_avg_class_recalls[class_idx]:.4f}')

    print('\n===============================================================\n')
    torch.save(best_model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', f'best_{args.model}_weights.pth'))

if __name__ == '__main__':
    main()