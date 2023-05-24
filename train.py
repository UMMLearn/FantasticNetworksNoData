from __future__ import print_function
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import datetime
from datetime import datetime
import numpy as np
import copy


from torchvision import datasets, transforms, models
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights, convnext_tiny, ConvNeXt_Tiny_Weights, densenet121, DenseNet121_Weights
from torch.hub import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url
#from torch.utils.tensorboard import SummaryWriter

#from models.wideresnet import *
from models.resnet import *
#from models.simple import *
#from models.densenet import *
#from models.resnext import *
#from models.allconv import *
#from models.wideresnet import *
from loss_utils import *
from utils import *

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# import augmentations
# from color_jitter import *
# from diffeomorphism import *
# from rand_filter import *

# from torch.distributions import Dirichlet, Beta
# from einops import rearrange, repeat
# from opt_einsum import contract

#from utils_confusion import *
#from utils_augmix import *
#from utils_prime import *
#from trades import trades_loss
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import L2DeepFoolAttack
#from create_data import compute_smooth_data, merge_data, CustomDataSet

#from robustness.datasets import CustomImageNet
#from robustness.datasets import DATASETS, DataSet, CustomImageNet
#import smoothers


parser = argparse.ArgumentParser(description='PyTorch CIFAR + proximity training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model',default="ResNet18",
                    help='network to use')
parser.add_argument('--dataset', default="CIFAR10",
                    help='which dataset to use, CIFAR10, CIFAR100, TINYIN')
parser.add_argument('--anneal', default="cosine", 
                    help='type of LR schedule stairstep, cosine, or cyclic')
parser.add_argument('--grad-clip', default = 1, type=int,
                    help='clip model weight gradients to 0.5')
parser.add_argument('--runs', default=5, type=int,
                    help='number of random intializations of prototypes')
parser.add_argument('--image-step', default=0.1, type=float,
                    help='learning rate for prototype image update')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='base learning rate')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--channel-norm', type=int, default=0,
                    help='whether to use specific whitening transforms per channel')




args = parser.parse_args()

kwargsUser = {}


# settings
if (args.model == "ResNet18"):
    network_string = "ResNet18"
else:
    print ("Invalid model architecture")
    
def get_datetime():
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H_%M_%S")
    return dt_string


with open('commandline_args.txt', 'a') as f:
    json.dump(args.__dict__, f, indent=2)
f.close()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}
torch.cuda.empty_cache()

print ("cuda: ", use_cuda)



def train_image_nodata(args, model, device, epoch, par_images, targets, iterations=200, transformDict={}, **kwargs):

    model.multi_out=1
    model.eval()

    image_lr = args.image_step

    for batch_idx in range(iterations):

        _par_images_opt = par_images.clone().detach().requires_grad_(True).to(device)

        _par_images_opt_norm = transformDict['norm'](_par_images_opt)

        L2_img, logits_img = model(_par_images_opt_norm)

        loss = F.cross_entropy(logits_img, targets, reduction='none')

        loss.backward(gradient=torch.ones_like(loss))

        with torch.no_grad():
            gradients_unscaled = _par_images_opt.grad.clone()
            grad_mag = gradients_unscaled.view(gradients_unscaled.shape[0], -1).norm(2, dim=-1)
            image_gradients = image_lr*gradients_unscaled  / grad_mag.view(-1, 1, 1, 1)  

            if (torch.mean(loss)>1e-7):
                par_images.add_(-image_gradients)

            par_images.clamp_(0.0,1.0)

            _par_images_opt.grad.zero_()

    model.multi_out=0

    return loss

def train(args, model, device, cur_loader, optimizer, epoch, scheduler=0.0, max_steps = 0, transformDict={}, **kwargs):


    print ('Training model')


    for batch_idx, (data, target) in enumerate(cur_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        
        data_norm = transformDict['norm'](data)

        Z = model(data_norm)
        loss = F.cross_entropy(Z, target)

        loss.backward()

        if (args.grad_clip):
            nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

        optimizer.step()

        if args.anneal == "cosine":
            if batch_idx < max_steps:
                scheduler.step()
                
        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(cur_loader.dataset),
                       100. * batch_idx / len(cur_loader), loss.item()))


def eval_train(args, model, device, train_loader, transformDict):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
        #for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = transformDict['norm'](data)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(args, model, device, test_loader, transformDict):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
        #for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = transformDict['norm'](data)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= (0.5*args.epochs):
        lr = args.lr * 0.1
    if epoch >= (0.75*args.epochs):
        lr = args.lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    MEAN = [0.5]*3
    STD = [0.5]*3
    
    if (args.dataset == "CIFAR10"):
        if args.channel_norm:
            MEAN = [0.4914, 0.4822, 0.4465]
            STD = [0.2471, 0.2435, 0.2616] 
    elif(args.dataset == "CIFAR100"):
        if args.channel_norm:
            MEAN = [0.5071, 0.4865, 0.4409]
            STD = [0.2673, 0.2564, 0.2762]
    elif (args.dataset == "TINYIN"):
        if args.channel_norm:
            MEAN = [0.4802, 0.4481, 0.3975]
            STD  = [0.2302, 0.2265, 0.2262]
    else:
        print ("ERROR dataset not found")


    if args.dataset in ["CIFAR10","CIFAR100"]:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])

        gen_transform_test = transforms.Compose(
            [transforms.ToTensor()])

        
    elif args.dataset in ["TINYIN"]:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4)])

        gen_transform_test = transforms.Compose(
            [transforms.ToTensor()])

    else:
        print ("ERROR setting transforms")


    
    if (args.dataset == "CIFAR10"):


        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)

        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=gen_transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        kwargsUser['num_classes'] = 10
        nclass=10
        nchannels = 3
        H, W = 32, 32
            
    elif (args.dataset == "CIFAR100"):

        trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)

        testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=gen_transform_test)

        kwargsUser['num_classes'] = 100
        nclass=100
        nchannels = 3
        H, W = 32, 32


    
    elif (args.dataset == "TINYIN"):
        

        trainset = datasets.ImageFolder(
            './tiny-imagenet-200/train',
            transform=train_transform)
        
        testset = datasets.ImageFolder(
            './tiny-imagenet-200/val/images',
            transform=gen_transform_test)

        kwargsUser['num_classes'] = 200
        nclass = 200
        nchannels = 3
        H, W = 64, 64

    else:
          
        print ("Error getting dataset")

    transformDict = {}
    transformDict['norm'] = transforms.Compose([transforms.Normalize(MEAN, STD)])

    splits = []
    all_inds = np.arange(len(trainset.targets))
    
    inds_train1, inds_test1, y_train1, y_test1 = train_test_split(all_inds, trainset.targets, test_size=0.25, random_state=42, stratify=trainset.targets)
    splits.append(inds_test1)

    inds_train2, inds_test2, y_train2, y_test2 = train_test_split(all_inds, trainset.targets, test_size=0.4, random_state=42, stratify=trainset.targets)
    splits.append(inds_test2)

    inds_train3, inds_test3, y_train3, y_test3 = train_test_split(all_inds, trainset.targets, test_size=0.6, random_state=42, stratify=trainset.targets)
    splits.append(inds_test3)

    inds_train4, inds_test4, y_train4, y_test4 = train_test_split(all_inds, trainset.targets, test_size=0.7, random_state=42, stratify=trainset.targets)
    splits.append(inds_test4)

    inds_train5, inds_test5, y_train5, y_test5 = train_test_split(all_inds, trainset.targets, test_size=0.8, random_state=42, stratify=trainset.targets)
    splits.append(inds_test5)

    inds_train6, inds_test6, y_train6, y_test6 = train_test_split(all_inds, trainset.targets, test_size=0.9, random_state=42, stratify=trainset.targets)
    splits.append(inds_test6)

    #add 100% training
    splits.append(all_inds)


    for j in range(len(splits)):

        # with open('train_hist.txt', 'a') as f:
        #     f.write("\n")
        #     f.write("Training: {} ".format(j))
        #     f.write("\n")
        #     f.write("TrainAcc \t TestAcc \t TrainLoss \t TestLoss \n")
        # f.close()

        subtrain = torch.utils.data.Subset(trainset, splits[j])
        print (len(subtrain))


        print('------------training no---------{}----------------------'.format(j))


        cur_loader = torch.utils.data.DataLoader(subtrain, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)

        if j==0:
            # init model, ResNet18() can be also used here for training
            if (args.model == "ResNet18"):
                if args.dataset in ["CIFAR10","CIFAR100"]:
                    model = ResNet18(nclass = nclass, scale=1.0, channels=nchannels, **kwargsUser).to(device)
                    #image_model = ResNet18(nclass = nclass,**kwargsUser).to(device)
                elif args.dataset in ["TINYIN"]:
                    model = ResNet18Tiny(nclass=nclass, scale=1.0, channels=nchannels, **kwargsUser).to(device)
                else:
                    print ("Error matching model to dataset")
            else:
                print ("Invalid model architecture")



            with torch.no_grad():
                prototype_batches = []
                #last_losses = []
                Mg = []
                Madv = []

                targets_onehot = torch.arange(nclass, dtype=torch.long, device=device)
                total_runs = args.runs

                for tr in range(total_runs):
                    #last_losses.append(0.0)
                    par_images_glob = torch.rand([nclass,nchannels,H,W],dtype=torch.float,device=device)
                    par_images_glob.clamp_(0.0,1.0)
                    prototype_batches.append(par_images_glob.clone().detach())


        if args.anneal in ["stairstep", "cosine"]:
            lr_i = args.lr
        else:
            print ("Error setting learning rate")

        optimizer = optim.SGD(cur_model.parameters(), lr=lr_i, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

        scheduler = 0.0
        steps_per_epoch = int(np.ceil(len(cur_loader.dataset) / args.batch_size))


        if args.anneal == "stairstep":
            pass
        elif args.anneal == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(cur_loader), eta_min=0.0000001, last_epoch=-1, verbose=False)
        else:
            print ("ERROR making scheduler") 

        #start training
        for epoch in range(1, args.epochs + 1):

            # adjust learning rate for SGD
            if args.anneal == "stairstep":
                adjust_learning_rate(optimizer, epoch)


            cur_model.multi_out = 0
            cur_model.train()
            
            train(args, cur_model, device, cur_loader, optimizer, epoch, scheduler=scheduler, max_steps = steps_per_epoch, transformDict=transformDict, **kwargsUser)


            cur_model.eval()

            print('================================================================')
            loss_train, acc_train = eval_train(args, cur_model, device, cur_loader, transformDict)
            loss_test, acc_test = eval_test(args, cur_model, device, test_loader, transformDict)
            print('================================================================')

            
            if (epoch == args.epochs):
                torch.save(cur_model.state_dict(),'model-{}-epoch{}-training{}.pt'.format(network_string,epoch,j))
                #torch.save(par_images_glob, os.path.join(model_dir,'prototypes_online_lyr_{}_pool_{}_epoch{}_training{}.pt'.format(args.proto_layer,args.proto_pool,epoch,j)))

        #Data Independent Assessment of Latest Model
        cur_model.eval()

        last_losses = []
        
        for run in range(total_runs):
            #par_images_glob_ref = par_images_class.clone().detach()
            last_loss = train_image_nodata(args, cur_model, device, epoch, prototype_batches[run], targets=targets_onehot,iterations=250, transformDict=transformDict, **kwargsUser)
            last_losses.append(torch.mean(last_loss).clone())
            #print ("final image loss: ", torch.mean(last_loss).item())


        
        cos_matrix_means = []
        df_latent_means = []

        for proto in prototype_batches:

            #Mg----------------------------------------------------------
            cur_model.multi_out=1
            proto_image_norm = transformDict['norm'](proto)

            latent_p, logits_p = cur_model(proto_image_norm)

            #compute cosine similarity matrix
            cos_mat_latent_temp = torch.zeros(nclass, nclass, dtype=torch.float)
            cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

            for i in range(len(latent_p)):
                for q in range(len(latent_p)):
                    if i != q:
                        cos_mat_latent_temp[i,q] = cos_sim(latent_p[i].view(-1), latent_p[q].view(-1))
                        #cos_mat_latent[i,q] = torch.sum((latent_p[i].view(-1))*(latent_p[q].view(-1)))

            #cos_matrices.append(cos_mat_latent_temp.clone())
            cos_matrix_means.append(1.0-torch.mean(cos_mat_latent_temp))

            
            #Madv----------------------------------------------------------
            cur_model.eval()
            cur_model.multi_out=0
            attack = L2DeepFoolAttack(overshoot=0.02)
            preprocessing = dict(mean=MEAN, std=STD, axis=-3)
            fmodel = PyTorchModel(cur_model, bounds=(0,1), preprocessing=preprocessing)

            raw, X_new_torch, is_adv = attack(fmodel, proto, targets_onehot, epsilons=100)

            cur_model.multi_out=1
            with torch.no_grad():
                X_new_torch_norm = transformDict['norm'](X_new_torch)
                latent_p_adv, logits_p_adv = cur_model(X_new_torch_norm)

                CS_df_latent = F.cosine_similarity(latent_p_adv.view(nclass,-1), latent_p.view(nclass,-1))
                df_latent_means.append(1.0-torch.mean(CS_df_latent))


        print ("Run Summary\n")
        print('================================================================\n')
        print ("Train Accuracy: {0:4.3f}\n".format(acc_train))
        print ("Test Accuracy: {0:4.3f}\n".format(acc_test))


        loss_mean = torch.mean(torch.stack(last_losses,dim=0),dim=0)
        print ("Mean Prototype Xent Loss: {0:4.6f}\n".format(loss_mean))

        cos_mat_std, cos_mat_moms = torch.std_mean(torch.stack(cos_matrix_means,dim=0),dim=0)
        print ("Mg Mean: {0:4.3f}".format(cos_mat_moms.item()))
        print ("Mg Mean Std: {0:4.3f}".format(cos_mat_std.item()))
        df_std, df_moms = torch.std_mean(torch.stack(df_latent_means,dim=0),dim=0)
        print ("Madv Mean: {0:4.3f}".format(df_moms.item()))
        print ("Madv Mean Std: {0:4.3f}".format(df_std.item()))


if __name__ == '__main__':
    main()
