# -*- coding: utf-8 -*-
# @Time   : 2023/4/27
# @Author : Haichao Zhang
# @Email  : Haichao.Zhang22@student.xjtlu.edu.cn
import os.path

import torch
import torch.nn as nn
from recbole.model.abstract_recommender import DebiasedRecommender

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
import numpy as np
from torch.autograd import Function

class DCCL_LGN(DebiasedRecommender):
    input_type = InputType.PAIRWISE
    BATCH_SIZE = 512
    neg_num = BATCH_SIZE
    NEG_IID_LIST_LEN = 200

    def __init__(self, config, dataset):
        super(DCCL_LGN, self).__init__(config, dataset)
        self.pop_coeff = 1.0
        self.score_coeff = 1.0
        self.dccl_int_weight = 1e-1
        self.dccl_conf_weight = 1e-1

        # load parameters info
        self.embedding_size = config["embedding_size"]

        # define layers and loss
        self.uid_int_emb_layer = nn.Embedding(self.n_users, self.embedding_size, padding_idx=0)
        self.uid_conf_emb_layer = nn.Embedding(self.n_users, self.embedding_size, padding_idx=0)

        self.iid_cont_emb_layer = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.iid_pop_emb_layer = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        self.apply(xavier_normal_initialization)
        base_model_file = "saved/LightGCN-Dec-05-2023_22-39-00.pth"
        if config["dataset"] == "ml-1m" and os.path.exists(base_model_file):
            print("base on LightGCN, dataset ml-1m")
            base_model = torch.load(base_model_file)
            self.uid_int_emb_layer.weight = torch.nn.Parameter(base_model["state_dict"]["user_embedding.weight"].data)
            self.uid_conf_emb_layer.weight = torch.nn.Parameter(base_model["state_dict"]["user_embedding.weight"].data)
            self.iid_pop_emb_layer.weight = torch.nn.Parameter(base_model["state_dict"]["item_embedding.weight"].data)
            self.iid_cont_emb_layer.weight = torch.nn.Parameter(base_model["state_dict"]["item_embedding.weight"].data)

        base_model_file = "saved/LightGCN-Dec-08-2023_02-50-38.pth"
        if config["dataset"] == "netflix" and os.path.exists(base_model_file):
            print("base on LightGCN, dataset netflix")
            base_model = torch.load(base_model_file)
            self.uid_int_emb_layer.weight = torch.nn.Parameter(base_model["state_dict"]["user_embedding.weight"].data)
            self.uid_conf_emb_layer.weight = torch.nn.Parameter(base_model["state_dict"]["user_embedding.weight"].data)
            self.iid_cont_emb_layer.weight = torch.nn.Parameter(base_model["state_dict"]["item_embedding.weight"].data)
            self.iid_pop_emb_layer.weight = torch.nn.Parameter(base_model["state_dict"]["item_embedding.weight"].data)

        base_model_file = "saved/LightGCN-Dec-08-2023_02-03-27.pth"
        if config["dataset"] == "amazon-luxury-beauty-18" and os.path.exists(base_model_file):
            print("base on LightGCN, dataset amazon-luxury-beauty-18")
            base_model = torch.load(base_model_file)
            self.uid_int_emb_layer.weight = torch.nn.Parameter(base_model["state_dict"]["user_embedding.weight"].data)
            self.uid_conf_emb_layer.weight = torch.nn.Parameter(base_model["state_dict"]["user_embedding.weight"].data)
            self.iid_cont_emb_layer.weight = torch.nn.Parameter(base_model["state_dict"]["item_embedding.weight"].data)
            self.iid_pop_emb_layer.weight = torch.nn.Parameter(base_model["state_dict"]["item_embedding.weight"].data)

    def pop_func(self, pop_tensor, pop_coeff):
        pop_tensor = torch.multiply(pop_tensor, pop_coeff)
        pop_tensor = torch.where(pop_tensor >= 1.0, torch.ones_like(pop_tensor), pop_tensor)
        pop_curve = torch.exp(-pop_tensor)
        # mask = tf.where(pop_tensor > pop_threshold, tf.ones_like(pop_tensor), pop_curve)
        return pop_curve

    def forward(self, interaction):
        pos_index = torch.where(interaction["label"] == 1)[0]
        neg_index = torch.where(interaction["label"] == 0)[0]

        uids = interaction[self.USER_ID][pos_index]
        iids = interaction[self.ITEM_ID][pos_index]
        neg_iid_list = interaction[self.ITEM_ID][neg_index].to(self.device)

        pop_indices = torch.nonzero(torch.logical_and(interaction["popular"] == 1, interaction["label"] == 1)).squeeze()
        item_pop_list = interaction[self.ITEM_ID][pop_indices]
        item_pop_max = max(item_pop_list)
        item_pop_norm_tensor = torch.tensor([e / item_pop_max for e in item_pop_list])

        uid_int_emb = self.uid_int_emb_layer(uids)
        uid_conf_emb = self.uid_int_emb_layer(uids)
        user_emb = torch.cat([uid_int_emb, uid_conf_emb], dim=-1)

        uids_1 = torch.reshape(uids, (-1, 1))
        uids_2 = torch.reshape(uids, (1, -1))
        y_true = torch.equal(uids_1, uids_2)

        # instance-level
        item_cont_emb = self.iid_cont_emb_layer(iids)
        item_pop_emb = self.iid_pop_emb_layer(iids)
        item_cont_emb_norm = torch.norm(item_cont_emb, p=2, dim=-1)
        item_pop_emb_norm = torch.norm(item_pop_emb, p=2, dim=-1)

        pos_item_pop = item_pop_norm_tensor

        mask_item_cont = self.pop_func(pos_item_pop, self.pop_coeff)
        mask_item_pop = torch.ones_like(mask_item_cont) - mask_item_cont

        uid_conf_emb = self.uid_conf_emb_layer(uids)
        uid_int_emb = self.uid_int_emb_layer(uids)
        user_int_emb_norm = torch.norm(uid_int_emb, p=2, dim=-1).to(self.device)
        user_conf_emb_norm = torch.norm(uid_conf_emb, p=2, dim=-1).to(self.device)

        item_pop_emb_norm = item_pop_emb_norm.unsqueeze(0).to(self.device)
        ui_int_score = torch.matmul(user_int_emb_norm, torch.transpose(item_pop_emb_norm,0,1))
        ui_int_score = ui_int_score * self.score_coeff

        item_pop_1 = torch.reshape(pos_item_pop, [-1, 1])
        item_pop_2 = torch.reshape(pos_item_pop, [1, -1])
        pop_select = torch.greater_equal(item_pop_1, item_pop_2).float().clone().detach().to(self.device)
        ui_conf_score = torch.matmul(user_conf_emb_norm, torch.transpose(item_pop_emb_norm,0,1)) * pop_select
        ui_conf_score = ui_conf_score * self.score_coeff

        iids_1 = torch.reshape(iids, [-1, 1])
        iids_2 = torch.reshape(iids, [1, -1])
        iid_eq = torch.tensor(torch.equal(iids_1, iids_2), dtype=torch.float32)
        y_true = (torch.tensor(y_true, dtype=torch.float32) * iid_eq).float() > 0.5

        ui_int_loss = self.CSELoss(ui_int_score, y_true, mask_item_cont, neg_num=self.neg_num)
        ui_conf_loss = self.CSELoss(ui_conf_score, y_true, mask_item_pop, neg_num=self.neg_num)

        rown = user_emb.size(0)
        r = torch.randint(low=0, high=(neg_iid_list.size(0) - 1), size=(rown,), dtype=torch.int64).to(self.device)
        r = r.view(rown, 1)
        r = r.view(-1)
        neg_iids = torch.gather(neg_iid_list, 0, r).to(self.device)

        neg_iids = neg_iids.view(rown)

        pos_iids_cont_emb = self.iid_cont_emb_layer(iids)
        pos_iids_pop_emb = self.iid_pop_emb_layer(iids)
        neg_iids_cont_emb = self.iid_cont_emb_layer(neg_iids)
        neg_iids_pop_emb = self.iid_pop_emb_layer(neg_iids)
        pos_iids_emb = torch.cat([pos_iids_cont_emb, pos_iids_pop_emb], dim=-1)
        neg_iids_emb = torch.cat([neg_iids_cont_emb, neg_iids_pop_emb], dim=-1)
        pos_score = torch.sum(user_emb * pos_iids_emb, dim=-1, keepdim=True)
        neg_score = torch.sum(user_emb * neg_iids_emb, dim=-1, keepdim=True)
        loss_total = torch.sum(-torch.log(torch.sigmoid(pos_score - neg_score) + 1e-9))

        return self.dccl_int_weight * ui_int_loss + self.dccl_conf_weight * ui_conf_loss + \
               loss_total
    def CSELoss(self, y_pred, label=None, mask=None, neg_num=128):
        if (label is None):
            n = y_pred.size(0)
            y_true = torch.eye(n, dtype=torch.float32)
        else:
            y_true = torch.tensor(label).float().clone().detach()
        N = y_pred.size(0)
        y_pred = torch.exp(y_pred)
        ratio = 1.0 - (neg_num + 1) * 1.0 / self.BATCH_SIZE
        pos_pred = torch.multiply(y_pred, y_true)
        ner = torch.sum(pos_pred, dim=-1, keepdim=False) + 1e-1
        if (ratio > 1e-5):
            mask = torch.greater(torch.empty([N, N],dtype=torch.float32).uniform_(), ratio).to(self.device)

            mask = torch.logical_or(mask, label).to(self.device)
            mask = torch.tensor(mask, dtype=torch.float32).to(self.device)
            der_pred = torch.multiply(y_pred, mask)
            der = torch.sum(torch.tensor(der_pred), dim=-1,  keepdim=False) + 1e-1
        else:
            der = torch.sum(y_pred, dim=-1, keepdim=False) + 1e-1
        loss = -torch.log(torch.div(ner, der) + 1e-12).to(self.device)
        if mask is None:
            pass
        else:
            mask = mask.to(loss.device)
            loss = loss * mask
        loss = torch.sum(loss)
        if torch.isnan(loss):
            return 0
        return loss

    def calculate_loss(self, interaction):
        loss = self.forward(interaction)
        return loss
    def predict(self, interaction):
        return self.full_sort_predict(interaction)


    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.uid_int_emb_layer(user)
        all_item_e = self.iid_cont_emb_layer.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

