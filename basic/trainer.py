import logging
import time
_logger = logging.getLogger('train')
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(model, dataloader, criterion, optimizer, log_interval, device):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    end = time.time()

    model.train()
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(dataloader):
        data_time_m.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)

        # predict
        outputs = model(inputs) #output: (b, class) = (4,10)
        loss = criterion(outputs, targets)
        loss.backward()

        # loss update
        optimizer.step()
        optimizer.zero_grad()
        losses_m.update(loss.item())

        # accuracy
        preds = outputs.argmax(dim=1)
        acc_m.update(targets.eq(preds).sum().item()/targets.size(0), n=targets.size(0))

        batch_time_m.update(time.time() - end)

def train_model(model, trainloader, valloader, criterion,optimizer, scheduler,epochs,savedir,log_interval,
        device):
    best_acc = 0
    step = 0

    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        train_metrics = train_epoch(model, trainloader, criterion, optimizer, log_interval, device)

