import sys
sys.path.append('../../../InfoSieve/')

import argparse
import os

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
from models import vision_transformer as vits


from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from tqdm import tqdm

from torch.nn import functional as F

from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path
from matplotlib import pyplot as plt
from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans

from kmeans_pytorch import kmeans

# TODO: Debug
import warnings
import math
warnings.filterwarnings("ignore")


class NormPLoss(torch.nn.Module):
    def __init__(self,pnorm=1, code_weight=None):
        super(NormPLoss, self).__init__()
        self.pnorm=pnorm
        self.code_weight=code_weight

    def forward(self, x):
        if self.code_weight is None:
            b = torch.norm(x, p=self.pnorm)
        else:
            b = torch.norm(x / self.code_weight, p=self.pnorm)/torch.norm(1/self.code_weight, p=self.pnorm)
        return b.mean()


class Norm0Loss(torch.nn.Module):
    def __init__(self):
        super(Norm0Loss, self).__init__()

    def forward(self, x):
        b=(x>0).float()
        return b.mean()


class HLoss(torch.nn.Module):
    def __init__(self,is_mask=False):
        super(HLoss, self).__init__()
        self.is_mask=is_mask

    def forward(self, x):
        if self.is_mask: b = ((1 - x) * x) ** 2
        else: b=(1-x**2)+x.mean(0)**2
        return b.mean()



class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None,is_code=False):#, smoothing=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        if is_code:
            dist = torch.cdist(anchor_feature, contrast_feature)
            dist=-dist/(dist.sum(dim=1)+1e-10)
        else:
            dist = -torch.cdist(anchor_feature, contrast_feature)

        anchor_dot_contrast = torch.div(dist, self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def info_nce_logits(features, args,is_code=False):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    if is_code:
        dist=torch.cdist(features, features,p=2)
        similarity_matrix =-dist/(dist.sum(dim=1)+1e-10)
    else:
        similarity_matrix=-torch.cdist(features, features)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels


def train(labeller_head, projection_head, model, train_loader, test_loader, unlabelled_train_loader, merge_train_loader, args):

    optimizer = SGD(list(labeller_head.parameters()) +list(projection_head.parameters()) + list(model.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    sup_con_crit = SupConLoss()
    sup_con_crit_codes = SupConLoss()
    L0loss = Norm0Loss()
    CatLoss = torch.nn.BCEWithLogitsLoss()
    Boundary_Loss_m = HLoss(is_mask=True)
    Boundary_Loss_z = HLoss()
    Linfloss = NormPLoss(pnorm=1, code_weight=labeller_head.code_weight)

    best_epoch_lab, best_epoch_comb, best_epoch = 0, 0, 0


    best_stats = []
    best_stats_proj=[]
    best_stats_label=[]
    Total_loss=[]
    Contrastive_loss=[]
    contrastive_code_loss=[]
    code_entropy=[]
    mask_entropy=[]
    code_length=[]
    category_loss=[]
    code_condition=[]
    mask_condition=[]
    accuracy_old=[]
    accuracy_new=[]
    accuracy_all=[]

    accuracy_old_comb=[]
    accuracy_new_comb=[]
    accuracy_all_comb=[]

    accuracy_old_code=[]
    accuracy_new_code=[]
    accuracy_all_code=[]

    old_acc_label_test, best_test_acc_lab_head = 0, 0
    old_acc_proj_test, best_test_acc_lab_comb = 0, 0
    old_acc_test, best_test_acc_lab = 0, 0
    all_acc, old_acc, new_acc = 0, 0, 0
    all_acc_test, old_acc_test, new_acc_test = 0, 0, 0
    all_acc_proj, old_acc_proj, new_acc_proj = 0, 0, 0
    all_acc_label, old_acc_label, new_acc_label = 0, 0, 0

    alpha = args.alpha  # The usual loss
    beta = args.beta  # The Code loss
    delta = args.delta  # LInfLoss
    mu = args.mu  # m cond Lossf
    eta = args.eta  # catloss
    zeta = args.zeta  # z cond Loss
    unsupervised_smoothing = args.unsupervised_smoothing
    train_report_interval = args.train_report_interval
    mytraining=args.mytraining

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        loss_cons_record = AverageMeter()

        with torch.no_grad():
            if mytraining:
                print('Extracting cluster labels unlabelled examples in the training data...')
                uq_index, all_preds, all_preds_lab ,metrics= test_sskmeans(model, merge_train_loader,
                                                                   projection_head=projection_head,
                                                                   labeller_head=labeller_head, args=args)
            else:
                print('Extracting cluster labels unlabelled examples in the training data...')
                uq_index, all_preds, metrics = test_sskmeans(model, merge_train_loader,args=args)

        if mytraining:
            loss_codes_record = AverageMeter()
            loss_zero_record = AverageMeter()
            loss_zero_mask_record = AverageMeter()

            loss_inf_record = AverageMeter()
            loss_cat_record = AverageMeter()

            loss_z_record = AverageMeter()
            loss_m_record = AverageMeter()

            labeller_head.age = (epoch+1)/10
            labeller_head.code_weight = torch.pow(2-epoch/args.epochs, torch.arange(0, -args.code_length, -1, dtype=torch.float32)).to(device)

        projection_head.train()
        model.train()
        labeller_head.train()

        for batch_idx, batch in enumerate(tqdm(train_loader)):

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
            images = torch.cat(images, dim=0).to(device)

            # Extract features with base model
            features = model(images)

            # Pass features through projection head
            features = projection_head(features)

            # L2-normalize features
            features = torch.nn.functional.normalize(features, dim=-1)

            # Choose which instances to run the contrastive loss on
            if args.contrast_unlabel_only:
                # Contrastive loss only on unlabelled instances
                f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)
            else:
                # Contrastive loss for all examples
                con_feats = features

            contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, args=args)
            contrastive_loss = torch.nn.CrossEntropyLoss(label_smoothing=unsupervised_smoothing)(contrastive_logits, contrastive_labels)

            pseudolabels = all_preds[np.argsort(uq_index)[uq_idxs]]
            pseudolabels[mask_lab] = class_labels[mask_lab]

            f1n, f2n = features.chunk(2)
            sup_feats = torch.cat([f1n.unsqueeze(1), f2n.unsqueeze(1)], dim=1)
            sup_labels = pseudolabels

            # Supervised contrastive loss
            f1, f2 = [f[mask_lab] for f in features.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)

            sup_con_loss+=sup_con_crit(sup_feats,labels=sup_labels)

            # Total loss
            loss_in = (1 - args.sup_con_weight) * contrastive_loss + args.sup_con_weight * sup_con_loss/2

            # Train acc
            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()
            train_acc_record.update(acc, pred.size(0))

            loss_cons_record.update(loss_in.item(), class_labels.size(0))

            """My Masker + Labeller Part"""
            #Give positional values to each place in features
            if mytraining:
                code_feat, mask, bin_raw, code, trunc_code = labeller_head(features)

                loss0= L0loss(bin_raw)
                loss0_mask = L0loss(mask)

                Length_loss = Linfloss(mask)
                Code_cond = Boundary_Loss_z(bin_raw)
                Mask_cond = Boundary_Loss_m(mask)

                # Supervised Category loss
                f1, f2 = [f[mask_lab] for f in trunc_code.chunk(2)]
                sup_feats = torch.cat([f1, f2], dim=0)
                sup_labels = (torch.nn.functional.one_hot(class_labels[mask_lab], args.num_labeled_classes) * 1.0).repeat(2, 1)
                catloss = CatLoss(labeller_head.categorizer(sup_feats), sup_labels)

                contrastive_logits_codes, contrastive_labels_codes = info_nce_logits(features=bin_raw, args=args, is_code=True)
                contrastive_loss_codes = torch.nn.CrossEntropyLoss(label_smoothing=unsupervised_smoothing)(contrastive_logits_codes, contrastive_labels_codes)

                # Supervised contrastive loss
                f1c, f2c = [f[mask_lab] for f in code.chunk(2)]
                sup_con_feats_codes = torch.cat([f1c.unsqueeze(1), f2c.unsqueeze(1)], dim=1)
                sup_con_labels_codes = class_labels[mask_lab]

                f1p, f2p = code.chunk(2)
                semisup_con_feats = torch.cat([f1p.unsqueeze(1), f2p.unsqueeze(1)], dim=1)
                sup_con_loss_codes = sup_con_crit_codes(sup_con_feats_codes, labels=sup_con_labels_codes, is_code=True)
                sup_con_loss_codes += sup_con_crit_codes(semisup_con_feats, labels=pseudolabels, is_code=True)

                loss_code =(1-args.sup_code_weight) * contrastive_loss_codes + (args.sup_code_weight) * sup_con_loss_codes/2

                loss_codes_record.update(loss_code.item(), class_labels.size(0))
                loss_zero_record.update(loss0.item(), class_labels.size(0))
                loss_zero_mask_record.update(loss0_mask.item(), class_labels.size(0))
                loss_inf_record.update(Length_loss.item(), class_labels.size(0))
                loss_cat_record.update(catloss.item(), class_labels.size(0))
                loss_z_record.update(Code_cond.item(), class_labels.size(0))
                loss_m_record.update(Mask_cond.item(), class_labels.size(0))

                # Total loss
                loss = alpha*loss_in + beta*loss_code + zeta*Code_cond + eta*catloss+ delta*Length_loss + mu*Mask_cond
            else:
                loss = loss_in

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch: {} Avg Loss: {:.3f} | Constrastive: {:.3f} '.format(epoch, loss_record.avg, loss_cons_record.avg))
        if mytraining:
            print('Code: {:.3f}| Code on: {:.3f} | Length Loss: {:.3f}'
                  '| Cat: {:.3f}| Code_condition: {:.3f}| Mask_Condition {:.3f}'.format(loss_codes_record.avg, loss_zero_record.avg
                                                                                , loss_inf_record.avg,loss_cat_record.avg
                                                                                , loss_z_record.avg, loss_m_record.avg ))
            Total_loss.append(loss_record.avg)
            Contrastive_loss.append(loss_cons_record.avg)
            contrastive_code_loss.append(loss_codes_record.avg)
            code_entropy.append(loss_zero_record.avg)
            code_length.append(loss_inf_record.avg)
            category_loss.append(loss_cat_record.avg)
            code_condition.append(loss_z_record.avg)
            mask_condition.append(loss_m_record.avg)

        with torch.no_grad():
            if not mytraining:
                if (epoch+1) % train_report_interval == 0:
                    print('Testing on unlabelled examples in the training data...')
                    all_acc, old_acc, new_acc = test_kmeans(model, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled',args=args)
                else:
                    all_acc, old_acc, new_acc = metrics["all_acc"], metrics["old_acc"], metrics["new_acc"]
                print('Testing on disjoint test set...')
                all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, test_loader, epoch=epoch, save_name='Test ACC', args=args)

            else:
                if (epoch+1) % train_report_interval == 0:
                    print('Testing on unlabelled examples in the training data...')
                    all_acc, old_acc, new_acc, all_acc_proj, old_acc_proj, new_acc_proj, all_acc_label, old_acc_label, new_acc_label  = \
                        test_kmeans(model, unlabelled_train_loader, projection_head=projection_head, labeller_head=labeller_head,
                                                            epoch=epoch, save_name='Train ACC Unlabelled proj head', args=args)
                else:
                    all_acc, old_acc, new_acc = metrics["all_acc"], metrics["old_acc"], metrics["new_acc"]
                    all_acc_label, old_acc_label, new_acc_label = metrics["all_acc_lab"], metrics["old_acc_lab"], metrics["new_acc_lab"]
                print('Testing on disjoint test set with projection head...')
                all_acc_test, old_acc_test, new_acc_test, all_acc_proj_test, old_acc_proj_test, new_acc_proj_test,\
                    all_acc_label_test, old_acc_label_test, new_acc_label_test = test_kmeans(model, test_loader,
                                                                       projection_head=projection_head,labeller_head=labeller_head,
                                                                       epoch=epoch, save_name='Test ACC proj head', args=args)
                accuracy_old.append(old_acc)
                accuracy_new.append( new_acc)
                accuracy_all.append(all_acc)

                accuracy_old_comb.append(old_acc_proj)
                accuracy_new_comb.append(new_acc_proj)
                accuracy_all_comb.append(all_acc_proj)

                accuracy_old_code.append(old_acc_label)
                accuracy_new_code.append(new_acc_label)
                accuracy_all_code.append(all_acc_label)

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        if (epoch+1) % train_report_interval == 0:
            print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

            if mytraining:
                print('Train Accuracies projection head: All {:.4f} | Old {:.4f} | New {:.4f}'
                      .format(all_acc_proj, old_acc_proj, new_acc_proj))
                print('Test Accuracies projection head: All {:.4f} | Old {:.4f} | New {:.4f}'
                      .format(all_acc_proj_test, old_acc_proj_test, new_acc_proj_test))
                print('Train Accuracies label head: All {:.4f} | Old {:.4f} | New {:.4f}'
                      .format(all_acc_label, old_acc_label, new_acc_label))
                print('Test Accuracies label head: All {:.4f} | Old {:.4f} | New {:.4f}'
                      .format(all_acc_label_test, old_acc_label_test, new_acc_label_test))

        # Step schedule
        exp_lr_scheduler.step()

        torch.save(model.state_dict(), args.model_path)
        print("model saved to {}.".format(args.model_path))

        torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')
        torch.save(labeller_head.state_dict(), args.model_path[:-3] + '_label_head.pt')

        if old_acc_test > best_test_acc_lab:

            print(f'Best ACC on new Classes on disjoint test set: {new_acc_test:.4f}...')
            #print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            best_stats =[all_acc,old_acc,new_acc]
            torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
            print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

            torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')
            torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_label_head_best.pt')

            best_test_acc_lab = old_acc_test
            best_epoch = epoch

        if old_acc_proj_test > best_test_acc_lab_comb:
            best_stats_proj =[all_acc_proj,old_acc_proj,new_acc_proj]

            torch.save(model.state_dict(), args.model_path[:-3] + f'comb_best.pt')
            print("model saved to {}.".format(args.model_path[:-3] + f'comb_best.pt'))

            torch.save(projection_head.state_dict(), args.model_path[:-3] + f'comb_proj_head_best.pt')
            torch.save(projection_head.state_dict(), args.model_path[:-3] + f'comb_label_head_best.pt')

            best_test_acc_lab_comb = old_acc_proj_test
            best_epoch_comb = epoch

        if old_acc_label_test > best_test_acc_lab_head:
            best_stats_label =[all_acc_label,old_acc_label,new_acc_label]
            torch.save(model.state_dict(), args.model_path[:-3] + f'lab_best.pt')
            print("model saved to {}.".format(args.model_path[:-3] + f'lab_best.pt'))

            torch.save(projection_head.state_dict(), args.model_path[:-3] + f'lab_proj_head_best.pt')
            torch.save(projection_head.state_dict(), args.model_path[:-3] + f'lab_label_head_best.pt')

            best_test_acc_lab_head = old_acc_label_test
            best_epoch_lab = epoch
        print('Best Train Epochs:  {} | Combined {} | Labels {}'.format(best_epoch, best_epoch_comb, best_epoch_lab))



    print('############# Final Reports #############')
    print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(best_stats[0], best_stats[1], best_stats[2]))
    if mytraining:
        print('Best Train Accuracies Combined: All {:.4f} | Old {:.4f} | New {:.4f}'
              .format(best_stats_proj[0], best_stats_proj[1], best_stats_proj[2]))
        print('Best Train Accuracies Labels: All {:.4f} | Old {:.4f} | New {:.4f}'
              .format(best_stats_label[0], best_stats_label[1], best_stats_label[2]))
        print('Best Train Epochs:  {} | Combined {} | Labels {}'
              .format(best_epoch, best_epoch_comb, best_epoch_lab))
        plot_a_loss(np.arange(0,int(args.epochs)),np.array(Total_loss), "Total")
        plot_a_loss(np.arange(0,int(args.epochs)),np.array(Contrastive_loss), "Contrastive")
        plot_a_loss(np.arange(0,int(args.epochs)),np.array(contrastive_code_loss), "Contrastive_Codes")
        plot_a_loss(np.arange(0,int(args.epochs)),np.array(category_loss), "Category")
        plot_a_loss(np.arange(0,int(args.epochs)),np.array(code_length), "Code_length")
        plot_a_loss(np.arange(0,int(args.epochs)),np.array(code_condition), "Code_condition")
        plot_a_loss(np.arange(0,int(args.epochs)),np.array(code_entropy), "Code_on")
        plot_a_loss(np.arange(0,int(args.epochs)),np.array(mask_condition), "Mask_condition")

        plot_acc(np.arange(0,int(args.epochs)),accuracy_old, accuracy_new, accuracy_all, "features")
        plot_acc(np.arange(0,int(args.epochs)),accuracy_old_code, accuracy_new_code, accuracy_all_code, "codes")
        plot_acc(np.arange(0,int(args.epochs)),accuracy_old_comb, accuracy_new_comb, accuracy_all_comb, "combination")


def plot_a_loss(x, y, name):

    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure(figsize=(10, 10))
    plt.title(name+" Loss")
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.plot(x, y, 'cornflowerblue',  linewidth=3)
    plt.xlim(0, x.shape[0]-1)
    plt.locator_params(axis='y', nbins=10)
    plt.xlabel("epochs",)
    plt.ylabel(" Loss")
    plt.savefig("Plots/"+name+"_Loss.png")


def plot_acc(x, old, new, all, name):
    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure(figsize=(10, 10))
    plt.title("Accuracy using "+name+" for clustering")
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.plot(x, old, color= 'cornflowerblue', label='Old', linewidth=3)
    plt.plot(x, new, color='orangered', label='New', linewidth=3)
    plt.plot(x, all, color='limegreen', label='All', linewidth=3)
    plt.xlabel("epochs")
    plt.ylabel("Accuray")
    plt.xlim(0,x.shape[0]-1)
    plt.locator_params(axis='both', nbins=10)
    plt.legend()
    plt.savefig("Plots/Accuracy_"+name+".png")


def test_sskmeans(model, test_loader, args,projection_head=None, labeller_head=None):
    model.eval()

    all_feats = []
    all_feats_comb = []
    all_feats_lab = []
    targets = np.array([])
    mask = np.array([])
    ids=np.array([])
    mask_cls=np.array([])
    metrics=dict()

    for batch_idx, (images, label, uq_idx, mask_lab_) in enumerate(tqdm(test_loader)):

        images = images[0].cuda()
        label, mask_lab_ = label.to(device), mask_lab_.to(device).bool()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        all_feats.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        if projection_head is not None:
            feats1 = projection_head(feats)
            if labeller_head is not None:
                feats2 = labeller_head(feats1)[2]
                feats_comb = torch.cat([feats, feats2], dim=1)
                feats_comb = torch.nn.functional.normalize(feats_comb, dim=-1)
                all_feats_comb.append(feats_comb.cpu().numpy())
                all_feats_lab.append(feats2.cpu().numpy())

        targets = np.append(targets, label.cpu().numpy())
        ids=np.append(ids,uq_idx.cpu().numpy())
        mask = np.append(mask, mask_lab_.cpu().bool().numpy())
        mask_cls = np.append(mask_cls,np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
    mask = mask.astype(bool)
    mask_cls = mask_cls.astype(bool)
    # -----------------------
    # K-MEANS
    # -----------------------
    # print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    l_feats = all_feats[mask]  # Get labelled set
    u_feats = all_feats[~mask]  # Get unlabelled set
    l_targets = targets[mask]  # Get labelled targets
    u_targets = targets[~mask]  # Get unlabelled targets
    if args.unbalanced: cluster_size=None
    else: cluster_size=math.ceil(len(targets)/(args.num_labeled_classes + args.num_unlabeled_classes))
    kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=10, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)

    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                              x in (l_feats, u_feats, l_targets, u_targets))

    kmeanssem.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeanssem.labels_
    mask_cls=mask_cls[~mask]
    preds = all_preds.cpu().numpy()[~mask]
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets.cpu().numpy(), y_pred=preds, mask=mask_cls,
                                                    eval_funcs=args.eval_funcs,
                                                    save_name='SS-K-Means Train ACC Unlabelled', print_output=True)
    metrics["all_acc"], metrics["old_acc"], metrics["new_acc"] = all_acc, old_acc, new_acc
    if projection_head is not None and labeller_head is not None:
        print('Fitting K-Means Labelled...')
        all_feats_lab = np.concatenate(all_feats_lab)
        l_feats = all_feats_lab[mask]  # Get labelled set
        u_feats = all_feats_lab[~mask]  # Get unlabelled set
        kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                                  max_iterations=10, init='k-means++',
                                  n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                                  mode=None, protos=None)
        l_feats, u_feats = (torch.from_numpy(x).to(device) for x in (l_feats, u_feats))

        kmeanssem.fit_mix(u_feats, l_feats, l_targets)
        all_preds_lab = kmeanssem.labels_
        preds = all_preds_lab.cpu().numpy()[~mask]
        all_acc_lab, old_acc_lab, new_acc_lab = log_accs_from_preds(y_true=u_targets.cpu().numpy(), y_pred=preds, mask=mask_cls,
                                                        eval_funcs=args.eval_funcs,
                                                        save_name='SS-K-Means Train ACC Unlabelled', print_output=True)
        metrics["all_acc_lab"], metrics["old_acc_lab"], metrics["new_acc_lab"] = all_acc_lab, old_acc_lab, new_acc_lab
        return ids, all_preds, all_preds_lab, metrics
    return ids,all_preds, metrics

def test_kmeans(model, test_loader, epoch,
                 save_name, args,projection_head=None, labeller_head=None, Use_GPU=True):

    model.eval()
    all_feats = []
    all_feats_comb = []
    all_feats_lab =[]
    targets = np.array([])
    mask = np.array([])

    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):

        images = images.cuda()
        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        all_feats.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        if projection_head is not None:
            feats1=projection_head(feats)
            if labeller_head is not None:
                feats2=labeller_head(feats1)[2]
                feats_comb=torch.cat([feats,feats2],dim=1)
                feats_comb = torch.nn.functional.normalize(feats_comb, dim=-1)
                all_feats_comb.append(feats_comb.cpu().numpy())
                all_feats_lab.append(feats2.cpu().numpy())

        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask.astype(bool)
    all_feats = np.concatenate(all_feats)
    # -----------------------
    # EVALUATE
    # -----------------------

    if Use_GPU:
        preds, prototypes = kmeans(X=torch.from_numpy(all_feats).to(device), num_clusters=args.num_unlabeled_classes+args.num_labeled_classes,
                                       distance='euclidean', device=device, tqdm_flag=False)

        preds, prototypes = preds.cpu().numpy(), prototypes.cpu().numpy()
    else:
        kmeanss = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(
            all_feats)
        preds = kmeanss.labels_

    #print('Done!')"""

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    if projection_head is not None and labeller_head is not None:
        print('Fitting K-Means Combined...')
        all_feats_comb = np.concatenate(all_feats_comb)
        if Use_GPU:
            preds, prototypes = kmeans(X=torch.from_numpy(all_feats).to(device),
                                       num_clusters=args.num_unlabeled_classes + args.num_labeled_classes,
                                       distance='euclidean', device=device, tqdm_flag=False)
            preds, prototypes = preds.cpu().numpy(), prototypes.cpu().numpy()
        else:
            kmeanss = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(
                all_feats_comb)
            preds = kmeanss.labels_
        all_acc_comb, old_acc_comb, new_acc_comb = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                        T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                        writer=args.writer)
        print('Fitting K-Means Labelled...')
        all_feats_lab = np.concatenate(all_feats_lab)
        if Use_GPU:
            preds, prototypes = kmeans(X=torch.from_numpy(all_feats).to(device),
                                       num_clusters=args.num_unlabeled_classes + args.num_labeled_classes,
                                       distance='euclidean', device=device, tqdm_flag=False)
            preds, prototypes = preds.cpu().numpy(), prototypes.cpu().numpy()
        else:
            kmeanss = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(
                all_feats_lab)
            preds = kmeanss.labels_
        all_acc_lab, old_acc_lab, new_acc_lab = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                        T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                        writer=args.writer)
        return all_acc, old_acc, new_acc, \
               all_acc_comb, old_acc_comb, new_acc_comb, \
               all_acc_lab, old_acc_lab, new_acc_lab

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, scars, aircraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)

    parser.add_argument('--grad_from_block', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)

    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--zeta', type=float, default=0.01)
    parser.add_argument('--eta', type=float, default=0.01)

    parser.add_argument('--sup_code_weight', type=float, default=0.35)
    parser.add_argument('--code_length', default=40, type=int)
    parser.add_argument('--unsupervised_smoothing', type=float, default=1)

    parser.add_argument('--train_report_interval', default=200, type=int)
    parser.add_argument('--gpu_clustering', type=str2bool, default=True)
    parser.add_argument('--unbalanced', type=str2bool, default=False)

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--mytraining', type=str2bool, default=True)
    parser.add_argument('--report', type=str2bool, default=True)



    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['metric_learn_gcd'])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model == 'vit_dino':

        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = dino_pretrain_path
        model = vits.__dict__['vit_base']()
        torch.cuda.empty_cache()

        state_dict = torch.load(pretrain_path, map_location='cpu')['teacher']
        dict_keys=list(state_dict.keys())
        for key in dict_keys:
            newkey= key.replace("backbone.",'')
            state_dict[newkey]=state_dict[key]
            del state_dict[key]
        model.load_state_dict(state_dict)

        if args.warmup_model_dir is not None:
            print(f'Loading weights from {args.warmup_model_dir}')
            model.load_state_dict(torch.load(args.warmup_model_dir+'model_best.pt', map_location='cpu'), strict=False)
        model.to(device)

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        max_block=0
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num>max_block:
                    max_block=block_num

                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    else:
        raise NotImplementedError
    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / (unlabelled_len+label_len) for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    merge_train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                               out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    if args.warmup_model_dir is not None:
        print(f'Loading projection head weights from {args.warmup_model_dir}')
        projection_head.load_state_dict(torch.load(args.warmup_model_dir + 'model_proj_head_best.pt', map_location='cpu'), strict=False)

    projection_head.to(device)

    code_weight = torch.pow(2, torch.arange(0, -args.code_length, -1, dtype=torch.float32)).to(device)
    labeller_head = vits.__dict__['DINOLABELLING'](in_dim=args.mlp_out_dim,
                                nlayers=2, bottleneck_dim= args.code_length, code_weight=code_weight, nclasses = args.num_labeled_classes)
    try:
        if args.warmup_model_dir is not None:
            print(f'Loading Labeller head weights from {args.warmup_model_dir}')
            labeller_head.load_state_dict(torch.load(args.warmup_model_dir + 'model_label_head_best.pt', map_location='cpu'), strict=False)
    except:
        pass

    labeller_head.to(device)
    # ----------------------
    # TRAIN
    # ----------------------
    if not os.path.exists('Plots'):
        os.mkdir('Plots')
    train(labeller_head, projection_head, model, train_loader, test_loader_labelled, test_loader_unlabelled, merge_train_loader, args)
    torch.cuda.empty_cache()
    if args.report:
        print("Reports for the best checkpoint:")
        os.system("CUDA_VISIBLE_DEVICES="+str(args.gpu_id)+" python ../clustering/extract_features.py --dataset "+args.dataset_name+
                  " --warmup_model_dir "+ args.model_path.replace('(','\(').replace(')','\)').replace('|','\|')+
                  " --mytraining "+str(int(args.mytraining)))
        os.system("CUDA_VISIBLE_DEVICES="+str(args.gpu_id)+" python ../clustering/k_means.py --dataset "+args.dataset_name+
                  " --mytraining "+str(int(args.mytraining))+" --unbalanced "+str(int(args.unbalanced)))
        print("Reports for the last checkpoint:")
        os.system("CUDA_VISIBLE_DEVICES="+str(args.gpu_id)+" python ../clustering/extract_features.py --dataset "+args.dataset_name+
                  " --warmup_model_dir "+ args.model_path.replace('(','\(').replace(')','\)').replace('|','\|')+
                  " --mytraining "+str(int(args.mytraining))+"  --use_best_model 0")
        os.system("CUDA_VISIBLE_DEVICES="+str(args.gpu_id)+" python ../clustering/k_means.py --dataset "+args.dataset_name+
                  " --mytraining "+str(int(args.mytraining))+" --unbalanced "+str(int(args.unbalanced)))

    torch.cuda.empty_cache()
    print(args.model_path)

