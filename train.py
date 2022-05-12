import os
import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.cifar import dataloader
from model.resnet import resnet
from utils.function import *


def train(args, net_student, net_teacher, train_loader, criterion, optimizer, global_step):
    for param in net_teacher.parameters():
        param.detach_()
    running_loss, processed_data = 0, 0
    correct, total = 0.0, 0.0
    net_student.train(), net_teacher.train()
    start = time.time()
    train_labeled = list(train_loader)[:(1000 // args.batch_size)]
    train_unlabeled = list(train_loader)[(1000 // args.batch_size):]
    train_labeled_iter = iter(train_labeled)
    train_unlabeled_iter = iter(train_unlabeled)
    for _ in tqdm.tqdm(range(len(train_unlabeled))):
        image_unlabeled, _ = next(train_unlabeled_iter)
        try:
            image_labeled, label_labeled = next(train_labeled_iter)
        except StopIteration:
            train_labeled_iter = iter(train_labeled)
            image_labeled, label_labeled = next(train_labeled_iter)
        image_labeled, label_labeled, image_unlabeled = Variable(image_labeled).cuda(), Variable(
            label_labeled).cuda(), Variable(image_unlabeled).cuda()
        images = torch.cat((image_labeled, image_unlabeled), dim=0)
        optimizer.zero_grad()
        out_student = net_student(images)
        with torch.no_grad():
            out_teacher = net_teacher(images)
        criterion_ent, criterion_mse, _, criterion_l1 = criterion
        loss_ce = criterion_ent(out_student[:image_labeled.shape[0]], label_labeled) / image_labeled.shape[0]
        # loss_cl = criterion_mse(F.softmax(out_student, dim=1), F.softmax(out_teacher, dim=1)) / image_labeled.shape[
        #     0] / args.num_classes
        # loss_cl = softmax_mse_loss(out_student, out_teacher) / images.shape[0]
        weight_cl = cal_consistency_weight(global_step, end_ep=(args.epochs // 2) * len(train_unlabeled), end_w=1.0)
        loss_cl = criterion_mse(out_student, out_teacher) / image_labeled.shape[0] / args.num_classes * weight_cl * 8.0
        loss_l1 = cal_reg_l1(net_student, criterion_l1)
        loss = loss_ce + loss_cl + loss_l1

        loss.backward()
        optimizer.step()
        update_ema_variables(net_student, net_teacher, alpha=0.95, global_step=global_step)
        global_step += 1

        running_loss += loss.item() * images.shape[0]
        processed_data += images.shape[0]
        correct += (out_teacher[:image_labeled.shape[0]].argmax(axis=1).cpu() == label_labeled.cpu()).sum()
        total += image_labeled.shape[0]
    print(loss_ce.item(), loss_cl.item(), loss_l1.item())
    train_loss = running_loss / processed_data
    accuracy = correct / total
    return global_step, train_loss, accuracy


def valid(args, net_teacher, test_loader, criterion):
    loss, correct, total = 0.0, 0.0, 0.0
    net_teacher.eval()
    criterion_ent, _, _, _ = criterion
    with torch.no_grad():
        for x, y in test_loader:
            x, y = Variable(x).cuda(), Variable(y).cuda()
            out = net_teacher(x).cuda()
            loss += criterion_ent(out, y).item()
            correct += (out.argmax(axis=1).cpu() == y.cpu()).sum()
            total += x.shape[0]
        return loss / total, correct / total


def main(args):
    train_loader, test_loader = dataloader(args)

    torch.backends.cudnn.enabled = True
    torch.manual_seed(7)
    net_student = resnet(depth=32, num_classes=args.num_classes).cuda()
    net_teacher = resnet(depth=32, num_classes=args.num_classes).cuda()

    criterion_ent = nn.CrossEntropyLoss(reduction='sum').cuda()
    criterion_mse = nn.MSELoss(reduction='sum').cuda()
    criterion_kl = nn.KLDivLoss(reduction='sum').cuda()
    criterion_l1 = nn.L1Loss(reduction='sum').cuda()
    criterion = (criterion_ent, criterion_mse, criterion_kl, criterion_l1)
    # optimizer = torch.optim.Adam(net_student.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(net_student.parameters(), lr=args.lr, momentum=0.9)
    global_step = 0
    for epoch in range(args.epochs):
        global_step, train_loss, train_accuracy = train(args, net_student, net_teacher, train_loader, criterion,
                                                        optimizer, global_step)
        valid_loss, valid_accuracy = valid(args, net_teacher, test_loader, criterion)
        print('train loss: {0} train accuracy: {1}'.format(train_loss, train_accuracy))
        print('valid loss: {0} valid accuracy: {1}'.format(valid_loss, valid_accuracy))
        adjust_learning_rate(args, optimizer, epoch)