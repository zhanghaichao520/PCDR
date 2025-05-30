# -*- coding: utf-8 -*-
# @Time   : 2023/4/27
# @Author : Haichao Zhang
# @Email  : Haichao.Zhang22@student.xjtlu.edu.cn
import os

import torch
import torch.nn as nn
from recbole.model.abstract_recommender import DebiasedRecommender

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
import numpy as np
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class PCDR_LGN(DebiasedRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(PCDR_LGN, self).__init__(config, dataset)
        # load parameters info
        self.embedding_size = 64
        self.weight0 = 2 # ML-1M
        # self.weight0 = 3 # Amazon-luxury-beauty-18
        # self.weight0 = 1 # Netflix
        self.weight1 = 0.5 # ML-1M
        # self.weight1 = 1 # Amazon-luxury-beauty-18
        # self.weight1 = 0.3 # Netflix
        self.weight2 = 2
        self.weight3 = 2
        # define layers and loss
        self.user_id_embedding = nn.Embedding(self.n_users, self.embedding_size)

        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self.loss = BPRLoss()

        self.matching_network = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_size * 2, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_size * 2, self.embedding_size),
            nn.ReLU()
        )
        self.conformity_network = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_size * 2, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_size * 2, self.embedding_size),
            nn.ReLU()
        )
        self.item_network = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_size * 2, self.embedding_size * 4),
            nn.ReLU(),
            nn.Linear(self.embedding_size * 4, self.embedding_size),
        )

        self.domain_classfier = nn.Sequential(
            nn.Linear(self.embedding_size, 1),
            nn.Sigmoid()
        )

        self.p = 0
        self.alpha = 2. / (1. + np.exp(-10 * self.p)) - 1
        # parameters initialization
        self.apply(xavier_normal_initialization)
        base_model_file = "saved/LightGCN-Dec-20-2023_15-38-56.pth"
        if config["dataset"] == "ml-1m" and os.path.exists(base_model_file):
            print("base on LightGCN, dataset ml-1m")
            base_model = torch.load(base_model_file)
            self.user_id_embedding.weight = torch.nn.Parameter(base_model["state_dict"]["user_embedding.weight"].data)
            self.item_id_embedding.weight = torch.nn.Parameter(base_model["state_dict"]["item_embedding.weight"].data)
        base_model_file = "saved/LightGCN-Apr-22-2025_00-56-11.pth"
        if config["dataset"] == "ml-100k" and os.path.exists(base_model_file):
            print("base on LightGCN, dataset ml-100k")
            base_model = torch.load(base_model_file)
            self.user_id_embedding.weight = torch.nn.Parameter(base_model["state_dict"]["user_embedding.weight"].data)
            self.item_id_embedding.weight = torch.nn.Parameter(base_model["state_dict"]["item_embedding.weight"].data)

        base_model_file = "saved/LightGCN-Dec-08-2023_02-50-38.pth"
        if config["dataset"] == "netflix" and os.path.exists(base_model_file):
            print("base on LightGCN, dataset netflix")
            base_model = torch.load(base_model_file)
            self.user_id_embedding.weight = torch.nn.Parameter(base_model["state_dict"]["user_embedding.weight"].data)
            self.item_id_embedding.weight = torch.nn.Parameter(base_model["state_dict"]["item_embedding.weight"].data)

        base_model_file = "saved/LightGCN-Dec-08-2023_02-03-27.pth"
        if config["dataset"] == "amazon-luxury-beauty-18" and os.path.exists(base_model_file):
            print("base on LightGCN, dataset amazon-luxury-beauty-18")
            base_model = torch.load(base_model_file)
            self.user_id_embedding.weight = torch.nn.Parameter(base_model["state_dict"]["user_embedding.weight"].data)
            self.item_id_embedding.weight = torch.nn.Parameter(base_model["state_dict"]["item_embedding.weight"].data)


        self.Ms_data = []
        self.Mt_data = []
        self.Cs_data = []
        self.Ct_data = []
    def get_user_embedding(self, interaction):
        id_embedding = self.user_id_embedding(interaction[self.USER_ID])
        return id_embedding
    def get_user_popular_embedding(self, interaction):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        popular_item0_embedding = self.item_id_embedding(interaction["item_interaction_popular0"])
        popular_item1_embedding = self.item_id_embedding(interaction["item_interaction_popular1"])
        popular_item2_embedding = self.item_id_embedding(interaction["item_interaction_popular2"])
        popular_item3_embedding = self.item_id_embedding(interaction["item_interaction_popular3"])
        popular_item4_embedding = self.item_id_embedding(interaction["item_interaction_popular4"])

        return self.get_user_embedding(interaction) + (popular_item0_embedding + popular_item1_embedding + popular_item2_embedding + popular_item3_embedding + popular_item4_embedding) / 5


    def get_user_unpopular_embedding(self, interaction):
        r"""Get a batch of user embedding tensor according to input user's id.
        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        unpopular_item0_embedding = self.item_id_embedding(interaction["item_interaction_unpopular0"])
        unpopular_item1_embedding = self.item_id_embedding(interaction["item_interaction_unpopular1"])
        unpopular_item2_embedding = self.item_id_embedding(interaction["item_interaction_unpopular2"])
        unpopular_item3_embedding = self.item_id_embedding(interaction["item_interaction_unpopular3"])
        unpopular_item4_embedding = self.item_id_embedding(interaction["item_interaction_unpopular4"])

        return self.get_user_embedding(interaction) + (unpopular_item0_embedding + unpopular_item1_embedding + unpopular_item2_embedding + unpopular_item3_embedding + unpopular_item4_embedding) / 5


    def get_item_embedding(self, interaction):
        r"""Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        id_embedding = self.item_id_embedding(interaction[self.ITEM_ID])
        return id_embedding
    def forward(self, interaction):
        dict = {}
        user_popular_embedding = self.get_user_popular_embedding(interaction)
        user_unpopular_embedding = self.get_user_unpopular_embedding(interaction)
        item_embedding = self.get_item_embedding(interaction)
        dict["user_popular_embedding"] = user_popular_embedding
        dict["user_unpopular_embedding"] = user_unpopular_embedding
        dict["item_embedding"] = item_embedding

        Ms = self.matching_network(user_popular_embedding)
        Mt = self.matching_network(user_unpopular_embedding)
        Cs = self.conformity_network(user_popular_embedding)
        Ct = self.conformity_network(user_unpopular_embedding)
        Iq = self.item_network(item_embedding)

        # save to dict
        dict["Ms"] = Ms
        dict["Mt"] = Mt
        dict["Cs"] = Cs
        dict["Ct"] = Ct
        dict["Iq"] = Iq

        Ms_Mt_simliar = torch.cosine_similarity(Ms, Mt)
        Ms_Cs_simliar = torch.cosine_similarity(Ms, Cs)
        Ms_Ct_simliar = torch.cosine_similarity(Ms, Ct)
        Mt_Cs_simliar = torch.cosine_similarity(Mt, Cs)
        Mt_Ct_simliar = torch.cosine_similarity(Mt, Ct)
        # save to dict
        dict["Ms_Mt_simliar"] = Ms_Mt_simliar
        dict["Ms_Cs_simliar"] = Ms_Cs_simliar
        dict["Ms_Ct_simliar"] = Ms_Ct_simliar
        dict["Mt_Cs_simliar"] = Mt_Cs_simliar
        dict["Mt_Ct_simliar"] = Mt_Ct_simliar

        Yd = interaction["popular"].unsqueeze(-1)

        sigmod = nn.Sigmoid()

        Y1 = ((Iq * ((Yd * Ms) + (1 - Yd) * Mt))).sum(dim = 1)
        Y2 = ((Iq * ((Yd * Cs) + (1 - Yd) * Ct))).sum(dim = 1)
        Y3 = ((Iq * ((Yd * (Ms + Cs)) + (1 - Yd) * (Mt + Ct)))).sum(dim = 1)

        dict["Y1_predict"] = Y1
        dict["Y2_predict"] = Y2
        dict["Y3_predict"] = Y3
        Y1 = sigmod(Y1)
        Y2 = sigmod(Y2)
        Y3 = sigmod(Y3)

        dict["Y1"] = Y1
        dict["Y2"] = Y2
        dict["Y3"] = Y3


        if self.training:
            if self.p < 1:
                self.p += 1 / 15000
                self.alpha = 2. / (1. + np.exp(-10 * self.p)) - 1

        domin_network_Ms = self.domain_classfier(ReverseLayerF.apply(Ms, self.alpha))
        domin_network_Mt = self.domain_classfier(ReverseLayerF.apply(Mt, self.alpha))
        domin_network_Cs = self.domain_classfier(Cs)
        domin_network_Ct = self.domain_classfier(Ct)
        dict["domin_network_Ms"] = domin_network_Ms
        dict["domin_network_Mt"] = domin_network_Mt
        dict["domin_network_Cs"] = domin_network_Cs
        dict["domin_network_Ct"] = domin_network_Ct

        return dict

    def calculate_loss(self, interaction):
        dict = self.forward(interaction)

        bce_loss = nn.BCELoss()
        bce_loss1 = bce_loss(dict["Y1"], interaction["label"])
        bce_loss2 = bce_loss(dict["Y2"], interaction["label"])
        bce_loss3 = bce_loss(dict["Y3"], interaction["label"])

        gm = 0.1
        sim_loss1 = torch.exp(dict["Ms_Mt_simliar"] / gm)
        sim_loss1 = sim_loss1.sum(dim = 0) / len(interaction)
        sim_loss1 = 1 / sim_loss1

        sim_loss2 = (torch.exp(dict["Ms_Cs_simliar"]) + torch.exp(dict["Ms_Ct_simliar"]) + torch.exp(dict["Mt_Cs_simliar"]) + torch.exp(dict["Mt_Ct_simliar"]))
        sim_loss2 = sim_loss2.sum(dim = 0) / len(interaction)

        causal_loss = torch.log(1 + torch.exp( (dict["Iq"] * dict["Ms"]).sum(dim=1) - (dict["Iq"] * dict["Mt"]).sum(dim=1)))
        causal_loss = causal_loss.sum(dim=0) / len(interaction)

        d0 = torch.zeros_like(dict["domin_network_Ms"])
        d1 = torch.ones_like(dict["domin_network_Ms"])

        domain_loss = bce_loss(dict["domin_network_Cs"], d1) + bce_loss(dict["domin_network_Ct"], d0) \
                      + bce_loss(dict["domin_network_Ms"], d1) + bce_loss(dict["domin_network_Mt"], d0)
        domain_loss = domain_loss / 4
        return ((bce_loss1 + bce_loss2 + bce_loss3) * self.weight0,  (sim_loss1 + sim_loss2) * self.weight1, causal_loss * self.weight2, domain_loss * self.weight3)


    def predict(self, interaction):
        dict = self.forward(interaction)
        return dict["Y1_predict"]

    def full_sort_predict(self, interaction):
        user_popular_embedding = self.get_user_popular_embedding(interaction)
        user_unpopular_embedding = self.get_user_unpopular_embedding(interaction)
        all_item_e = self.item_id_embedding.weight

        Yd = interaction["popular"].unsqueeze(-1)

        Ms = self.matching_network(user_popular_embedding)
        Mt = self.matching_network(user_unpopular_embedding)
        Cs = self.conformity_network(user_popular_embedding)
        Ct = self.conformity_network(user_unpopular_embedding)

        Y1 = torch.matmul(((Yd * Ms) + (1 - Yd) * Mt), all_item_e.transpose(0, 1))  # [user_num,item_num]
        Y2 = torch.matmul(((Yd * Cs) + (1 - Yd) * Ct), all_item_e.transpose(0, 1))  # [user_num,item_num]
        Y3 = torch.matmul(((Yd * (Ms + Cs)) + (1 - Yd) * (Mt + Ct)), all_item_e.transpose(0, 1))  # [user_num,item_num]

        return Y1.view(-1)

