import yaml
import argparse
import random
import numpy as np
import torch
import timm
# from model import VPT
from model import VIT_Base_Deep as VPT
from dataset import MyDataset, data_gather
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from trainer import train_model
import os

def run(cfg):

    savedir = os.path.join(cfg['RESULT']['savedir'],  cfg['EXP_NAME'])
    os.makedirs(savedir, exist_ok=True)

    # setup_default_logging(log_path=os.path.join(savedir,'log.txt'))

    #랜덤 시드 고정: 실험할 때 동일한 값 나올 수 있게
    random.seed(cfg['SEED']) #python set
    np.random.seed(cfg['SEED']) #numpy 라이브러리 set
    torch.manual_seed(cfg['SEED']) #pytorch set
    # torch.backends.cudnn.deterministic = True #cuddn 랜덤시드 고정시(true) => 단점: 속도 느려짐

    ##2. device 세팅
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = timm.create_model('vit_base_patch8_224', )

    ##3. dataset 세팅
    # data_list, label_list = Dataset(cfg['DATASET'])
    data_list, label_list = data_gather(cfg['DATASET'])


    ##3.1. train-val split
    train_lists, val_lists, train_labels, val_labels = train_test_split(data_list, label_list, train_size=0.8, shuffle=True, random_state=cfg['SEED'], stratify=label_list)
    #dataloader => mini batch 만들어주는 엳할
    trainloader = DataLoader(dataset= MyDataset(cfg['DATASET'],train_lists,train_labels), batch_size=cfg['TRAINING']['batch_size'], shuffle=True, num_workers=4)
    valloader = DataLoader(dataset= MyDataset(cfg['DATASET'],val_lists,val_labels), batch_size=cfg['TRAINING']['batch_size'], shuffle=True, num_workers=4)

    ##4. model 세팅
    model = VPT(
            modelname      = cfg['MODEL']['name'],
            num_classes    = cfg['DATASET']['num_classes'],
            pretrained     = True,
            prompt_tokens  = cfg['MODEL']['prompt_tokens'],
            prompt_dropout = cfg['MODEL']['prompt_dropout']
        )
    model.to(device)
    print('# of learnable params: {}'.format(np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))

    ##5.loss, optimizer, criterion, scheduler 세팅
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), cfg['OPTIMIZER']['params']['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['TRAINING']['epochs'])

    ##6. Train model
    train_model(
        model,
        trainloader  = trainloader,
        valloader   = valloader,
        criterion    = criterion,
        optimizer    = optimizer,
        scheduler    = scheduler,
        epochs       = cfg['TRAINING']['epochs'],
        savedir      = savedir,
        log_interval = cfg['TRAINING']['log_interval'],
        device       = device,
   )








if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Visual Prompt Tuning')
    parser.add_argument('--config_file', type=str, default='./config.yaml')

    args = parser.parse_args()
    cfg = yaml.load(open(args.config_file ,'r'), Loader=yaml.FullLoader)
    run(cfg)


