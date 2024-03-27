
import torch
import torch.nn.functional as F


def label_smooth_loss_fn(outputs, targets, epsilon=0.1):
    onehot = F.one_hot(targets, 10).float()
    targets = (1 - epsilon) * onehot + torch.ones(onehot.shape) * epsilon / 1000
    return loss_fn(outputs, targets)

def loss_fn(outputs, targets):
    if len(targets.shape) == 1:
        return F.cross_entropy(outputs, targets)
    else:
        return torch.mean(torch.sum(-targets * F.log_softmax(outputs, dim=1), dim=1))



if __name__=='__main__':
    data_root ='/Users/sungheui/PycharmProjects/basic/dataset/train'

    outputs = torch.rand(1,10)
    target = torch.tensor([3])

    lot = label_smooth_loss_fn(outputs, target)
