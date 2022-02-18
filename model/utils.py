import torch


# from scipy.optimize import linear_sum_assignment

def distloss(x, target):
    B, C, H, W = x.shape
    x = x.view(-1)
    target = target.view(-1)
    # row_ind, col_ind = linear_sum_assignment(cost.cpu().detach().numpy(), maximize=False)
    # loss = cost.gather(1, torch.from_numpy(col_ind).view(-1, 1).to(x.device)).mean()
    loss = ((x.sort()[0] - target.sort()[0]) ** 2).mean()
    return loss


if __name__ == '__main__':
    x = torch.tensor([0., 1., 2., 3.], requires_grad=True)
    loss = distloss(x, torch.tensor([0.1, 3.1, 2.1, 1.1]))
    loss.backward()
    print(loss, x.grad)
