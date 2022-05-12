import math
import torch
import bisect
import torch.nn.functional as functional


def update_ema_variables(model, model_teacher, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1.0 - 1.0 / float(global_step + 1), alpha)
    for param_t, param in zip(model_teacher.parameters(), model.parameters()):
        param_t.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = functional.softmax(input_logits, dim=1)
    target_softmax = functional.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return functional.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes


def cal_reg_l1(model, criterion_l1):
    reg_loss = 0
    np = 0
    for param in model.parameters():
        reg_loss += criterion_l1(param, torch.zeros_like(param))
        np += param.nelement()
    reg_loss = reg_loss / np
    return reg_loss


def cal_consistency_weight(epoch, init_ep=0, end_ep=150, init_w=0.0, end_w=20.0):
    """Sets the weights for the consistency loss"""
    if epoch > end_ep:
        weight_cl = end_w
    elif epoch < init_ep:
        weight_cl = init_w
    else:
        T = float(epoch - init_ep)/float(end_ep - init_ep)
        #weight_mse = T * (end_w - init_w) + init_w #linear
        weight_cl = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w #exp
    #print('Consistency weight: %f'%weight_cl)
    return weight_cl


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 at [150, 225, 300] epochs"""

    boundary = [args.epochs // 2, args.epochs // 4 * 3, args.epochs]
    lr = args.lr * 0.1 ** int(bisect.bisect_left(boundary, epoch))
    print('Learning rate: %f' % lr)
    # print(epoch, lr, bisect.bisect_left(boundary, epoch))
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr