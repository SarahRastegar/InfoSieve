import sys
sys.path.append('../../../InfoSieve/')

import torch
from torch.utils.data import DataLoader

import argparse
import os
from tqdm import tqdm
import numpy as np

from data.stanford_cars import CarsDataset
from data.cifar import CustomCIFAR10, CustomCIFAR100, cifar_10_root, cifar_100_root
from data.herbarium_19 import HerbariumDataset19, herbarium_dataroot
from data.augmentations import get_transform
from data.imagenet import get_imagenet_100_datasets
from data.data_utils import MergedDataset
from data.cub import CustomCub2011, cub_root
from data.fgvc_aircraft import FGVCAircraft, aircraft_root
from models import vision_transformer as vits
from data.pets import OxfordIIITPet, pets_root
from data.get_datasets import get_class_splits

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from project_utils.general_utils import strip_state_dict, str2bool
from copy import deepcopy
from torchvision.utils import save_image

from config import feature_extract_dir, dino_pretrain_path

chkpt='(08.01.2024_|_05.622)'

def plot_tsne(X,y,target_names):
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)

    target_ids = range(len(target_names))

    plt.figure(figsize=(6, 5))
    colors = 'r', 'orange', 'y', 'g', 'limegreen', 'c', 'b', 'purple', 'm','tab:pink'
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], s=5,c=c, label=label)
    plt.legend()
    plt.axis('off')
    plt.show()

def plot_hierarchy(X,y,target_names):
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)

    target_ids = range(len(target_names))

    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    plt.legend()
    plt.show()

def extract_features_dino(model, loader, save_dir,projection_head=None, labeller_head=None,
                          model_comb=None, projection_head_comb=None,
                          labeller_head_comb=None,
                          model_lab=None, projection_head_lab=None,
                          labeller_head_lab=None,
                          fold='train',silent=True):

    model.to(device)
    model.eval()

    all_feats = []
    all_feats_comb = []
    all_feats_lab = []

    comb_all_feats = []
    lab_all_feats = []
    targets = np.array([])

    if args.mytraining:
        model_comb.to(device)
        model_comb.eval()
        model_lab.to(device)
        model_lab.eval()
        projection_head_comb=projection_head_comb.to(device)
        projection_head_comb.eval()
        labeller_head_comb=labeller_head_comb.to(device)
        labeller_head_comb.eval()
        projection_head_lab=projection_head_lab.to(device)
        projection_head_lab.eval()
        labeller_head_lab=labeller_head_lab.to(device)
        labeller_head_lab.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, disable =silent)):

            images, labels, idxs = batch[:3]
            images = images.to(device)
            targets = np.append(targets, labels.cpu().numpy())

            features = model(images)         # CLS_Token for ViT, Average pooled vector for R50
            all_feats.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
            if args.mytraining:
                comb_features = model_comb(images)
                lab_features=model_lab(images)
                comb_all_feats.append(torch.nn.functional.normalize(comb_features, dim=-1).cpu().numpy())
                lab_all_feats.append(torch.nn.functional.normalize(lab_features, dim=-1).cpu().numpy())

                comb_feats1 = projection_head_comb(comb_features)
                comb_feats_lab = labeller_head_comb(comb_feats1)[2]

                lab_feats1 = projection_head_lab(lab_features)
                lab_feats_lab = labeller_head_lab(lab_feats1)[2]

                features_comb = torch.cat([comb_features, comb_feats_lab], dim=1)
                features_comb = torch.nn.functional.normalize(features_comb, dim=-1)
                all_feats_comb.append(features_comb.cpu().numpy())

                feats_lab = lab_feats_lab
                all_feats_lab.append(feats_lab.cpu().numpy())
            features = torch.nn.functional.normalize(features, dim=-1)

            # Save features

            if args.mytraining:
                save_dir_normal= os.path.join(save_dir , fold)
                save_dir_comb= os.path.join(save_dir + '/Comb', fold)
                save_dir_lab= os.path.join(save_dir + '/Labels', fold)

                for f,fcomb,flab, t, uq in zip(features,features_comb,feats_lab, labels, idxs):
                    t = t.item()
                    uq = uq.item()

                    save_path = os.path.join(save_dir_normal, f'{t}', f'{uq}.npy')
                    torch.save(f.detach().cpu().numpy(), save_path)

                    save_path = os.path.join(save_dir_comb, f'{t}', f'{uq}.npy')
                    torch.save(fcomb.detach().cpu().numpy(), save_path)

                    save_path = os.path.join(save_dir_lab, f'{t}', f'{uq}.npy')
                    torch.save(flab.detach().cpu().numpy(), save_path)
            else:
                save_dir_normal= os.path.join(save_dir , fold)

                for f, t, uq in zip(features, labels, idxs):
                    t = t.item()
                    uq = uq.item()

                    save_path = os.path.join(save_dir_normal, f'{t}', f'{uq}.npy')
                    torch.save(f.detach().cpu().numpy(), save_path)

def extract_features_timm(model, loader, save_dir):

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):

            images, labels, idxs = batch[:3]
            images = images.to(device)

            features = model.forward_features(images)         # CLS_Token for ViT, Average pooled vector for R50

            # Save features
            for f, t, uq in zip(features, labels, idxs):

                t = t.item()
                uq = uq.item()

                save_path = os.path.join(save_dir, f'{t}', f'{uq}.npy')
                torch.save(f.detach().cpu().numpy(), save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--root_dir', type=str, default=feature_extract_dir)
    parser.add_argument('--warmup_model_dir', type=str,
                        default='../contrastive_training/'
                                'osr_novel_categories/metric_learn_gcd/log/'+chkpt+'/checkpoints/model.pt')
    parser.add_argument('--use_best_model', type=str2bool, default=True)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset', type=str, default='scars', help='options: cifar10, cifar100, scars,aircraft')
    parser.add_argument('--silent', type=str2bool, default=True)
    parser.add_argument('--code_length', default=40, type=int)
    parser.add_argument('--mytraining', type=str2bool, default=True)



    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')

    args.save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset}')
    if not args.silent:
        print(args)
        print('Loading model...')
    else:
        import warnings
        warnings.filterwarnings("ignore")#, category=DeprecationWarning)
    # ----------------------
    # MODEL
    # ----------------------
    if args.model_name == 'vit_dino':
        extract_features_func = extract_features_dino
        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = dino_pretrain_path
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536
        args.dataset_name=args.dataset
        args = get_class_splits(args)
        args.num_labeled_classes = len(args.train_classes)
        args.num_unlabeled_classes = len(args.unlabeled_classes)

        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
        if args.mytraining:
            model_comb = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
            model_lab = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)

        state_dict = torch.load(pretrain_path, map_location='cpu')['teacher']
        dict_keys=list(state_dict.keys())
        for key in dict_keys:
            newkey= key.replace("backbone.",'')
            state_dict[newkey]=state_dict[key]
            del state_dict[key]
        model.load_state_dict(state_dict)
        if args.mytraining:
            model_comb.load_state_dict(state_dict)
            model_lab.load_state_dict(state_dict)

        _, val_transform = get_transform('imagenet', image_size=224, args=args)

    elif args.model_name == 'resnet50_dino':

        extract_features_func = extract_features_dino
        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = '/work/sagar/pretrained_models/dino/dino_resnet50_pretrain.pth'

        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', pretrained=False)

        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

        _, val_transform = get_transform('imagenet', image_size=224, args=args)

    else:

        raise NotImplementedError

    if args.warmup_model_dir is not None:

        warmup_id = args.warmup_model_dir.split('(')[1].split(')')[0]

        if args.use_best_model:
            warmup_dir= args.warmup_model_dir[:-3]
            args.warmup_model_dir = warmup_dir + '_best.pt'
            if args.mytraining:
                args.warmup_model_dir_comb = warmup_dir + 'comb_best.pt'
                args.warmup_model_dir_lab = warmup_dir + 'lab_best.pt'

        if not args.silent: print(f'Using weights from {args.warmup_model_dir} ...')
        state_dict = torch.load(args.warmup_model_dir)
        model.load_state_dict(state_dict)
        if args.mytraining:
            model_comb.load_state_dict(state_dict)
            model_lab.load_state_dict(state_dict)

        if not args.silent: print(f'Saving to {args.save_dir}')
        # ----------------------
        # PROJECTION HEAD
        # ----------------------
        projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                                    out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
        if args.mytraining:
            projection_head_comb = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                                        out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
            projection_head_lab = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                                        out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
        if args.warmup_model_dir is not None:
            if not args.silent: print(f'Loading projection head weights from {args.warmup_model_dir}')
            if args.use_best_model:
                projection_head.load_state_dict(
                torch.load(args.warmup_model_dir.split('model_best.pt')[0] + 'model_proj_head_best.pt', map_location='cpu'), strict=False)
                if args.mytraining:
                    projection_head_comb.load_state_dict(
                        torch.load(args.warmup_model_dir.split('model_best.pt')[0] + 'modelcomb_proj_head_best.pt',
                                   map_location='cpu'), strict=False)
                    projection_head_lab.load_state_dict(
                        torch.load(args.warmup_model_dir.split('model_best.pt')[0] + 'modellab_proj_head_best.pt',
                                   map_location='cpu'), strict=False)
            else:
                projection_head.load_state_dict(
                    torch.load(args.warmup_model_dir.split('model.pt')[0] + 'model_proj_head.pt',
                               map_location='cpu'), strict=False)

        projection_head.to(device)

        code_weight = torch.pow(2, torch.arange(0, -args.code_length, -1, dtype=torch.float32)).to(device)
        labeller_head = vits.__dict__['DINOLABELLING'](in_dim=args.mlp_out_dim,
                                                       nlayers=2, bottleneck_dim=args.code_length,
                                                       code_weight=code_weight, nclasses=args.num_labeled_classes)
        if args.mytraining:
            labeller_head_comb = vits.__dict__['DINOLABELLING'](in_dim=args.mlp_out_dim,
                                                           nlayers=2, bottleneck_dim=args.code_length,
                                                           code_weight=code_weight, nclasses=args.num_labeled_classes)
            labeller_head_lab = vits.__dict__['DINOLABELLING'](in_dim=args.mlp_out_dim,
                                                           nlayers=2, bottleneck_dim=args.code_length,
                                                           code_weight=code_weight, nclasses=args.num_labeled_classes)

        if args.warmup_model_dir is not None:
            if not args.silent: print(f'Loading Labeller head weights from {args.warmup_model_dir}')
            if args.use_best_model:

                labeller_head.load_state_dict(
                torch.load(args.warmup_model_dir.split('model_best.pt')[0] + 'model_label_head_best.pt', map_location='cpu'), strict=False)
                if args.mytraining:
                    labeller_head_comb.load_state_dict(
                    torch.load(args.warmup_model_dir.split('model_best.pt')[0] + 'modelcomb_label_head_best.pt',
                               map_location='cpu'), strict=False)
                    labeller_head_lab.load_state_dict(
                        torch.load(args.warmup_model_dir.split('model_best.pt')[0] + 'modellab_label_head_best.pt',
                                   map_location='cpu'), strict=False)
            else:
                labeller_head.load_state_dict(
                    torch.load(args.warmup_model_dir.split('model.pt')[0] + 'model_label_head.pt',
                               map_location='cpu'), strict=False)


        labeller_head.to(device)

    if not args.silent: print('Loading data...')
    # ----------------------
    # DATASET
    # ----------------------
    if args.dataset == 'cifar10':

        train_dataset = CustomCIFAR10(root=cifar_10_root, train=True, transform=val_transform)
        test_dataset = CustomCIFAR10(root=cifar_10_root, train=False, transform=val_transform)
        targets = list(set(train_dataset.targets))

    elif args.dataset == 'cifar100':

        train_dataset = CustomCIFAR100(root=cifar_100_root, train=True, transform=val_transform)
        test_dataset = CustomCIFAR100(root=cifar_100_root, train=False, transform=val_transform)
        targets = list(set(train_dataset.targets))

    elif args.dataset == 'scars':

        train_dataset = CarsDataset(train=True, transform=val_transform)
        test_dataset = CarsDataset(train=False, transform=val_transform)
        targets = list(set(train_dataset.target))
        targets = [i - 1 for i in targets]          # SCars are labelled 1 - 197. Change to 0 - 196

    elif args.dataset == 'herbarium_19':

        train_dataset = HerbariumDataset19(root=os.path.join(herbarium_dataroot, 'small-train'),
                                           transform=val_transform)

        test_dataset = HerbariumDataset19(root=os.path.join(herbarium_dataroot, 'small-validation'),
                                           transform=val_transform)

        targets = list(set(train_dataset.targets))

    elif args.dataset == 'imagenet_100':

        datasets = get_imagenet_100_datasets(train_transform=val_transform, test_transform=val_transform,
                                             train_classes=range(50),
                                             prop_train_labels=0.5)

        datasets['train_labelled'].target_transform = None
        datasets['train_unlabelled'].target_transform = None

        train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                      unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

        test_dataset = datasets['test']
        targets = list(set(test_dataset.targets))

    elif args.dataset == 'cub':

        train_dataset = CustomCub2011(root=cub_root, transform=val_transform, train=True)
        test_dataset = CustomCub2011(root=cub_root, transform=val_transform, train=False)
        targets = list(set(train_dataset.data.target.values))
        targets = [i - 1 for i in targets]          # SCars are labelled 1 - 200. Change to 0 - 199

    elif args.dataset == 'aircraft':

        train_dataset = FGVCAircraft(root=aircraft_root, transform=val_transform, split='trainval')
        test_dataset = FGVCAircraft(root=aircraft_root, transform=val_transform, split='test')
        targets = list(set([s[1] for s in train_dataset.samples]))

    elif args.dataset == 'pets':

        train_dataset = OxfordIIITPet(root=pets_root, transform=val_transform, split='trainval')
        test_dataset =  OxfordIIITPet(root=pets_root, transform=val_transform, split='test')
        targets = list(set([s[1] for s in train_dataset.samples]))

    else:

        raise NotImplementedError

    # ----------------------
    # DATALOADER
    # ----------------------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if not args.silent: print('Creating base directories...')
    # ----------------------
    # INIT SAVE DIRS
    # Create a directory for each class
    # ----------------------
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for fold in ('train', 'test'):
        fold_dir = os.path.join(args.save_dir, fold)
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)

        for t in targets:
            target_dir = os.path.join(fold_dir, f'{t}')
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

    if args.mytraining:
        if not os.path.exists(args.save_dir+'/Comb'):
            os.makedirs(args.save_dir+'/Comb')

        if not os.path.exists(args.save_dir+'/Labels'):
            os.makedirs(args.save_dir+'/Labels')
        for fsplit in ('/Comb','/Labels'):
            for fold in ('train', 'test'):
                fold_dir = os.path.join(args.save_dir+fsplit, fold)
                if not os.path.exists(fold_dir):
                    os.mkdir(fold_dir)

                for t in targets:
                    target_dir = os.path.join(fold_dir, f'{t}')
                    if not os.path.exists(target_dir):
                        os.mkdir(target_dir)

    # ----------------------
    # EXTRACT FEATURES
    # ----------------------
    # Extract train features


    if not args.silent: print('Extracting features from train split...')
    if args.mytraining:
        extract_features_func(model=model, loader=train_loader, save_dir=args.save_dir,projection_head=projection_head, labeller_head=labeller_head,
                              model_comb=model_comb,projection_head_comb=projection_head_comb, labeller_head_comb=labeller_head_comb,
                              model_lab=model_lab,projection_head_lab=projection_head_lab, labeller_head_lab=labeller_head_lab,fold='train')
    else:
        extract_features_func(model=model, loader=train_loader, save_dir=args.save_dir, projection_head=projection_head,
                              labeller_head=labeller_head, fold='train')

    # Extract test features
    if not args.silent: print('Extracting features from test split...')
    if args.mytraining:
        extract_features_func(model=model, loader=test_loader, save_dir=args.save_dir,projection_head=projection_head,labeller_head=labeller_head,
                          model_comb=model_comb,projection_head_comb=projection_head_comb, labeller_head_comb=labeller_head_comb,
                          model_lab=model_lab,projection_head_lab=projection_head_lab, labeller_head_lab=labeller_head_lab,fold='test')
    else:
        extract_features_func(model=model, loader=test_loader, save_dir=args.save_dir, projection_head=projection_head,
                              labeller_head=labeller_head, fold='test')