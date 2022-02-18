from multiprocessing import cpu_count

import torch
import torchvision.datasets
import torchvision.transforms as T

import wandb
from model import modeldic as M
from utils.utils import tensor2pilimg


def operate(phase):
    if phase == 'train':
        model.train()
        loader = trainloader
    else:
        model.eval()
        loader = valloader
    for idx, (data, target) in enumerate(loader):
        data = data.to(device)
        loss, out = model.trainbatch(data)
        if phase == 'train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if args.wandb:
            wandb.log({f'{phase}_idx': phaseidx[phase], **{f'{phase}/{k}':out['stats'][k] for k in out['stats']}, })
            if idx == 0:
                wandb.log(
                    {f'{phase}_idx': phaseidx[phase], f'{phase}/img': wandb.Image(tensor2pilimg(list(out['img'].values())))}
                )
        print(f'{epoch}/{args.epoch}, {idx}/{len(loader)}, {out["stats"]}, {phase}')
        phaseidx[phase] += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tags')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--model', default='vae')
    args = parser.parse_args()

    device = args.device
    if args.wandb:
        wandbrun = wandb.init(project='vae', tags=args.tags,
                              config={'model': args.model, 'batchsize': args.batchsize, 'epoch': args.epoch})
        wandbrun.define_metric('epoch')
        wandbrun.define_metric('train_idx')
        wandbrun.define_metric('val_idx')
    model = M[args.model]().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='../data', transform=T.Compose([T.ToTensor()])), batch_size=args.batchsize,
        shuffle=True, num_workers=cpu_count())
    valloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(train=False, transform=T.Compose([T.ToTensor()]), root='../data'),
        batch_size=args.batchsize, shuffle=True, num_workers=cpu_count())
    phaseidx = {"train": 0, "val": 0}
    for epoch in range(args.epoch):
        operate('train')
        operate('val')
