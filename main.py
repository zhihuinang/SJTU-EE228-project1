import argparse
import os
import torch
from tqdm import trange
from torchvision.transforms import transforms
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np

from info import INFO
from model import ResNet18,ResNet50
from utils import ACC,AUC
from dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
                    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal


def train(model,optimizer,loss,train_loader,device,task):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        (inputs,labels) = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if task == 'multi-label, binary-class':
            labels = labels.to(torch.float32).to(device)
            err = loss(outputs, labels)
        else:
            labels = labels.squeeze().long().to(device)
            err = loss(outputs, labels)
        err.backward()
        optimizer.step()

def val(model, val_loader, device, val_auc_list, task, dir_path, epoch):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                labels = labels.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                labels = labels.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                labels = labels.float().resize_(len(labels), 1)

            y_true = torch.cat((y_true, labels), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_pred = y_score.detach().cpu().numpy()
        auc = AUC(y_true, y_pred, task)
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }
    print('Finish train epoch {}, AUC:{}'.format(epoch,auc))
    path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)


def test(model, split, data_loader, device, flag, task, output_root=None):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = AUC(y_true, y_score, task)
        acc = ACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))

        # if output_root is not None:
        #     output_dir = os.path.join(output_root, flag)
        #     if not os.path.exists(output_dir):
        #         os.mkdir(output_dir)
        #     output_path = os.path.join(output_dir, '%s.csv' % (split))
        #     save_results(y_true, y_score, output_path)




def main(args):
    data_name = args.data_name.lower()
    input_root = args.input_root
    output_root = args.output_root
    num_epoch = args.num_epoch
    download = args.download
    model_type = args.model

    flag_to_class = {
        "pathmnist": PathMNIST,
        "chestmnist": ChestMNIST,
        "dermamnist": DermaMNIST,
        "octmnist": OCTMNIST,
        "pneumoniamnist": PneumoniaMNIST,
        "retinamnist": RetinaMNIST,
        "breastmnist": BreastMNIST,
        "organmnist_axial": OrganMNISTAxial,
        "organmnist_coronal": OrganMNISTCoronal,
        "organmnist_sagittal": OrganMNISTSagittal,
    }
    DataClass = flag_to_class[data_name]

    info = INFO[data_name]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    lr = 0.001
    batch_size = 128
    val_auc_list = []
    dir_path = os.path.join(output_root, '%s_checkpoints' % (data_name+"_"+model_type))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('doing data preprocessing......')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    train_dataset = DataClass(root=input_root,
                                    split='train',
                                    transform=transform,
                                    download=download)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)
    val_dataset = DataClass(root=input_root,
                                  split='val',
                                  transform=transform,
                                  download=download)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    test_dataset = DataClass(root=input_root,
                                   split='test',
                                   transform=transform,
                                   download=download)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    print('data preprocessing done.....')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_type == 'ResNet18':
        model = ResNet18(in_ch=n_channels,class_num=n_classes)
    else:
        model = ResNet50(in_ch=n_channels,class_num=n_classes)
    model = model.to(device)
    if task == 'multi-label, binary-class':
        loss = nn.BCEWithLogitsLoss()
    else:
        loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in trange(0, num_epoch):
        train(model, optimizer, loss, train_loader, device, task)
        val(model, val_loader, device, val_auc_list, task, dir_path, epoch)

    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    print('epoch %s is the best model' % (index))

    print('==> Testing model...')
    restore_model_path = os.path.join(
        dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    model.load_state_dict(torch.load(restore_model_path)['net'])
    test(model,
         'train',
         train_loader,
         device,
         data_name,
         task,
         output_root=output_root)
    test(model, 'val', val_loader, device, data_name, task, output_root=output_root)
    test(model,
         'test',
         test_loader,
         device,
         data_name,
         task,
         output_root=output_root)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST')
    parser.add_argument('--data_name',
                        default='pathmnist',
                        help='subset of MedMNIST',
                        type=str)
    parser.add_argument('--input_root',
                        default='../../data',
                        help='input root, the source of dataset files',
                        type=str)
    parser.add_argument('--output_root',
                        default='../output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epoch',
                        default=5,
                        help='num of epochs of training',
                        type=int)
    parser.add_argument('--download',
                        default=True,
                        help='whether download the dataset or not',
                        type=bool)
    parser.add_argument('--model',
                        default='ResNet18',
                        help='model type',
                        type=str)

    args = parser.parse_args()

    main(args)
