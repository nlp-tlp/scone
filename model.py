#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from tqdm import tqdm
#  import numpy as np

pi = 3.14159265358979323846
pi_eps = 0.001


def convert_to_arg(x):
    y = torch.tanh(2 * x) * pi / 2 + pi / 2
    return y


def convert_to_axis(x):
    y = torch.tanh(x) * pi
    return y


class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = pi
        return axis_embedding / self.embedding_range * scale


class Projection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, nrelation,
                 embedding_range, angle_scale, axis_scale,
                 arg_scale, proj_net, att_mode, num_rel_base=30):
        super(Projection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_range = embedding_range
        self.angle_scale = angle_scale
        self.axis_scale = axis_scale
        self.arg_scale = arg_scale
        self.n_base = num_rel_base
        self.proj_net = proj_net
        self.att_mode = att_mode

        # rtransform
        if (self.proj_net == "rtrans") | (self.proj_net == "rtrans_mlp"):
            self.rel_base = nn.Parameter(torch.zeros(
                self.n_base, self.relation_dim * 2, self.relation_dim * 2))
            self.rel_att = nn.Parameter(torch.zeros(nrelation, self.n_base))
            self.rel_bias = nn.Parameter(
                torch.zeros(self.n_base, self.relation_dim * 2))
            self.norm = nn.LayerNorm(
                self.relation_dim * 2, elementwise_affine=False)

            torch.nn.init.orthogonal_(self.rel_base)
            torch.nn.init.xavier_normal_(self.rel_bias)
            torch.nn.init.xavier_normal_(self.rel_att)

        # mlp
        if (self.proj_net == "mlp") | (self.proj_net == "rtrans_mlp"):
            self.rel_axis_embedding = nn.Parameter(torch.zeros(
                nrelation, self.relation_dim), requires_grad=True)
            nn.init.uniform_(
                tensor=self.rel_axis_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            self.rel_arg_embedding = nn.Parameter(torch.zeros(
                nrelation, self.relation_dim), requires_grad=True)
            nn.init.uniform_(
                tensor=self.rel_arg_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

            self.layer1 = nn.Linear(
                self.entity_dim + self.relation_dim, self.hidden_dim)
            self.layer0 = nn.Linear(
                self.hidden_dim, self.entity_dim + self.relation_dim)
            self.layer0_norm = nn.LayerNorm(
                self.entity_dim + self.relation_dim, elementwise_affine=False)
            for nl in range(2, num_layers + 1):
                setattr(self, "layer{}".format(nl), nn.Linear(
                    self.hidden_dim, self.hidden_dim))
            for nl in range(num_layers + 1):
                nn.init.xavier_uniform_(
                    getattr(self, "layer{}".format(nl)).weight)

        # attention
        if (self.proj_net == "rtrans_mlp"):
            self.layer_att1 = nn.Linear(2 * self.entity_dim, self.entity_dim)
            self.layer_att2 = nn.Linear(self.entity_dim, 2 * self.entity_dim)

            nn.init.xavier_uniform_(self.layer_att1.weight)
            nn.init.xavier_uniform_(self.layer_att2.weight)

    def forward(self, source_embedding_axis, source_embedding_arg, rel_indices):

        # rtransform
        if (self.proj_net == "rtrans") | (self.proj_net == "rtrans_mlp"):
            rel_att = self.rel_att[rel_indices]
            rel_att = self.angle_scale(rel_att, self.arg_scale)
            rel_att = convert_to_axis(rel_att)

            e_embedding = torch.cat(
                [source_embedding_axis, source_embedding_arg], dim=-1)
            project_r = torch.einsum('br,rio->bio', rel_att, self.rel_base)
            if self.rel_bias.shape[0] == self.rel_base.shape[0]:
                bias = torch.einsum('br,ri->bi', rel_att, self.rel_bias)
            else:
                bias = self.rel_bias[rel_indices]
            output = torch.einsum('bio,bi->bo', project_r, e_embedding) + bias
            x_rtransform = self.norm(output)

        # mlp
        if (self.proj_net == "mlp") | (self.proj_net == "rtrans_mlp"):
            r_embedding_axis = torch.index_select(
                self.rel_axis_embedding, dim=0, index=rel_indices)
            r_embedding_axis = self.angle_scale(
                r_embedding_axis, self.axis_scale)
            r_embedding_axis = convert_to_axis(r_embedding_axis)

            r_embedding_arg = torch.index_select(
                self.rel_arg_embedding, dim=0, index=rel_indices)
            r_embedding_arg = self.angle_scale(r_embedding_arg, self.arg_scale)
            r_embedding_arg = convert_to_arg(r_embedding_arg)

            output = torch.cat([source_embedding_axis + r_embedding_axis,
                                source_embedding_arg + r_embedding_arg], dim=-1)
            for nl in range(1, self.num_layers + 1):
                output = F.relu(getattr(self, "layer{}".format(nl))(output))
            output = self.layer0(output)
            x_mlp = self.layer0_norm(output)

        # attention
        if (self.proj_net == "rtrans_mlp"):
            x_stacked = []
            x_stacked.append(x_rtransform)
            x_stacked.append(x_mlp)
            x_att = torch.stack(x_stacked)
            '''
            (2, batch_size, relation_dim * 2)
            '''
            if self.att_mode == "stack":
                pass
            elif self.att_mode == "dot_prod":
                '''
                Q (batch_size, relation_dim * 2, 1)
                K (batch_size, 1, relation_dim * 2)
                '''
                x_att = torch.matmul(torch.unsqueeze(
                    x_rtransform, 2), torch.unsqueeze(x_mlp, 1))
                x_att = torch.permute(x_att, (1, 0, 2))
            elif self.att_mode == "scaled_dot_prod":
                x_att = torch.matmul(torch.unsqueeze(x_rtransform, 2), torch.unsqueeze(
                    x_mlp, 1)) / torch.sqrt(torch.tensor(x_rtransform.size()[1]))
                x_att = torch.permute(x_att, (1, 0, 2))

            layer1_act = F.relu(self.layer_att1(x_att))
            attention = F.softmax(self.layer_att2(layer1_act), dim=0)
            # merge
            x_merged = torch.sum(attention * x_att, dim=0)

        if (self.proj_net == "rtrans"):
            axis, arg = torch.chunk(x_rtransform, 2, dim=-1)
        elif (self.proj_net == "mlp"):
            axis, arg = torch.chunk(x_mlp, 2, dim=-1)
        elif (self.proj_net == "rtrans_mlp"):
            axis, arg = torch.chunk(x_merged, 2, dim=-1)

        axis_embeddings = convert_to_axis(axis)
        arg_embeddings = convert_to_arg(arg)

        return axis_embeddings, arg_embeddings


def intersection_cone(cone1, cone2, conj_mode, delta_none):
    '''
    num_cond = ["all", "partial", "complete", "none"]
    '''
    axis1, arg1 = cone1
    axis2, arg2 = cone2

    arg_int = torch.min(arg1, arg2)
    axis_int = torch.min(axis1, axis2)

    upper1 = (axis1 + arg1)
    lower1 = (axis1 - arg1)
    upper2 = (axis2 + arg2)
    lower2 = (axis2 - arg2)

    mask11 = (upper1 >= upper2) & (upper2 >= lower1) & (lower1 >= lower2)
    mask12 = (upper1 >= upper2) & (upper2 >= lower2) & (lower2 > lower1)
    mask13 = (upper1 >= lower1) & (lower1 > upper2) & (upper2 >= lower2)

    mask21 = (upper2 >= upper1) & (upper1 >= lower2) & (lower2 >= lower1)
    mask22 = (upper2 >= upper1) & (upper1 >= lower1) & (lower1 > lower2)
    mask23 = (upper2 >= lower2) & (lower2 > upper1) & (upper1 >= lower1)

    if conj_mode == "all":
        # aperture intersection
        arg_int[mask11] = torch.abs(upper2[mask11] - lower1[mask11]) * 0.5
        arg_int[mask12] = arg2[mask12]
        arg_int[mask13] = torch.zeros_like(arg_int[mask13])

        arg_int[mask21] = torch.abs(upper1[mask21] - lower2[mask21]) * 0.5
        arg_int[mask22] = arg1[mask22]
        arg_int[mask23] = torch.zeros_like(arg_int[mask23])

        # axis intersection
        axis_int[mask11] = upper2[mask11] - arg_int[mask11]
        axis_int[mask12] = axis2[mask12]
        axis_int[mask13] = delta_none * lower1[mask13] + \
            (1 - delta_none) * upper2[mask13]

        axis_int[mask21] = upper1[mask21] - arg_int[mask21]
        axis_int[mask22] = axis1[mask22]
        axis_int[mask23] = delta_none * lower2[mask23] + \
            (1 - delta_none) * upper1[mask23]

    elif conj_mode == "partial":
        # aperture intersection
        arg_int[mask11] = torch.abs(upper2[mask11] - lower1[mask11]) * 0.5
        arg_int[mask12] = torch.abs(upper2[mask12] - lower1[mask12]) * 0.5
        arg_int[mask13] = torch.abs(upper2[mask13] - lower1[mask13]) * 0.5

        arg_int[mask21] = torch.abs(upper1[mask21] - lower2[mask21]) * 0.5
        arg_int[mask22] = torch.abs(upper1[mask22] - lower2[mask22]) * 0.5
        arg_int[mask23] = torch.abs(upper1[mask23] - lower2[mask23]) * 0.5

        # axis intersection
        axis_int[mask11] = upper2[mask11] - arg_int[mask11]
        axis_int[mask12] = upper2[mask12] - arg_int[mask12]
        axis_int[mask13] = upper2[mask13] - arg_int[mask13]

        axis_int[mask21] = upper1[mask21] - arg_int[mask21]
        axis_int[mask22] = upper1[mask22] - arg_int[mask22]
        axis_int[mask23] = upper1[mask23] - arg_int[mask23]

    elif conj_mode == "complete":
        # aperture intersection
        arg_int[mask11] = arg2[mask11]
        arg_int[mask12] = arg2[mask12]
        arg_int[mask13] = arg2[mask13]

        arg_int[mask21] = arg1[mask21]
        arg_int[mask22] = arg1[mask22]
        arg_int[mask23] = arg1[mask23]

        # axis intersection
        axis_int[mask11] = axis2[mask11]
        axis_int[mask12] = axis2[mask12]
        axis_int[mask13] = axis2[mask13]

        axis_int[mask21] = axis1[mask21]
        axis_int[mask22] = axis1[mask22]
        axis_int[mask23] = axis1[mask23]

    elif conj_mode == "none":
        # aperture intersection
        arg_int = torch.zeros_like(arg_int)

        # axis intersection
        axis_int[mask11] = lower1[mask11] - \
            torch.abs(lower1[mask11] - upper2[mask11]) * 0.5
        axis_int[mask12] = lower1[mask12] - \
            torch.abs(lower1[mask12] - upper2[mask12]) * 0.5
        axis_int[mask13] = lower1[mask13] - \
            torch.abs(lower1[mask13] - upper2[mask13]) * 0.5

        axis_int[mask21] = lower2[mask21] - \
            torch.abs(lower2[mask21] - upper1[mask21]) * 0.5
        axis_int[mask22] = lower2[mask22] - \
            torch.abs(lower2[mask22] - upper1[mask22]) * 0.5
        axis_int[mask23] = lower2[mask23] - \
            torch.abs(lower2[mask23] - upper1[mask23]) * 0.5

    return axis_int, arg_int


class Intersection(nn.Module):
    def __init__(self, dim, logic_mode, conj_mode, delta_none):
        super(Intersection, self).__init__()
        self.dim = dim
        self.logic_mode = logic_mode
        self.conj_mode = conj_mode
        self.delta_none = delta_none
        if (self.logic_mode == "attention"):
            self.layer1 = nn.Linear(2*self.dim, 1*self.dim)
            self.layer2 = nn.Linear(1*self.dim, self.dim)

            nn.init.xavier_uniform_(self.layer1.weight)
            nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, axis_embeddings, arg_embeddings):
        if self.logic_mode == "attention":
            # (num_conj, batch_size, 2 * dim)
            all_embeddings = torch.cat(
                [axis_embeddings, arg_embeddings], dim=-1)
            # (num_conj, batch_size, 1 * dim)
            layer1_act = F.relu(self.layer1(all_embeddings))
            # (num_conj, batch_size, dim)
            attention = F.softmax(self.layer2(layer1_act), dim=0)

            axis_embeddings = torch.sum(attention * axis_embeddings, dim=0)
            arg_embeddings = torch.sum(attention * arg_embeddings, dim=0)
        elif self.logic_mode == "geometry":
            if axis_embeddings.size()[0] == 2:
                cone1 = (axis_embeddings[0, :, :], arg_embeddings[0, :, :])
                cone2 = (axis_embeddings[1, :, :], arg_embeddings[1, :, :])
                axis_embeddings, arg_embeddings = intersection_cone(
                    cone1, cone2, self.conj_mode, self.delta_none)
            elif axis_embeddings.size()[0] == 3:
                cone1 = (axis_embeddings[0, :, :], arg_embeddings[0, :, :])
                cone2 = (axis_embeddings[1, :, :], arg_embeddings[1, :, :])
                cone3 = (axis_embeddings[2, :, :], arg_embeddings[2, :, :])
                # (i+1) and (i+2) intersection
                cone12 = tuple(intersection_cone(
                    cone1, cone2, self.conj_mode, self.delta_none))
                # (i+1, i+2) and (i+3) intersection
                axis_embeddings, arg_embeddings = intersection_cone(
                    cone12, cone3, self.conj_mode, self.delta_none)

        return axis_embeddings, arg_embeddings


class Negation(nn.Module):
    def __init__(self):
        super(Negation, self).__init__()

    def forward(self, axis_embedding, arg_embedding):

        mask_pos = (axis_embedding >= 0.)
        axis_embedding[mask_pos] = axis_embedding[mask_pos] - pi
        axis_embedding[~mask_pos] = axis_embedding[~mask_pos] + pi

        arg_embedding = pi - arg_embedding

        return axis_embedding, arg_embedding


class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, emb_dim, gamma, geo, use_cuda,
                 proj_mode, query_name_dict, center_reg, logic_mode,
                 psi_axis_distance, distance_type, proj_net, conj_mode, att_mode, delta_none,
                 test_batch_size=1):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.epsilon = torch.tensor(2.0)
        self.geo = geo
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda(
        ) if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1)  # used in test_step
        self.query_name_dict = query_name_dict
        self.psi = psi_axis_distance
        self.distance_type = distance_type
        self.proj_net = proj_net
        self.conj_mode = conj_mode
        self.att_mode = att_mode
        self.delta_none = delta_none

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.cen = center_reg
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / emb_dim]),
            requires_grad=False
        )
        self.modulus = nn.Parameter(torch.Tensor(
            [0.5 * self.embedding_range.item()]), requires_grad=True)

        self.entity_dim = emb_dim
        self.relation_dim = emb_dim
        self.logic_mode = logic_mode
        hidden_dim, num_layers = proj_mode

        # scale axis embeddings to [-pi, pi]
        self.angle_scale = AngleScale(self.embedding_range.item())
        self.axis_scale = 1.0
        self.arg_scale = 1.0

        if self.geo == 'scone':
            self.entity_embedding = nn.Parameter(
                torch.zeros(nentity, self.entity_dim))
            self.negation = Negation()
            self.projection = Projection(self.entity_dim, hidden_dim, num_layers, self.nrelation,
                                         self.embedding_range, self.angle_scale, self.axis_scale,
                                         self.arg_scale, self.proj_net, self.att_mode)
            self.intersection = Intersection(
                self.entity_dim, self.logic_mode, self.conj_mode, self.delta_none)

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def forward(self, positive_sample, negative_sample, subsampling_weight,
                batch_queries_dict, batch_idxs_dict):
        if self.geo == 'scone':
            return self.forward_model(positive_sample, negative_sample,
                                      subsampling_weight, batch_queries_dict, batch_idxs_dict)

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]  # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2],
                                            queries[:, 5:6]], dim=1),
                                 torch.cat([queries[:, 2:4],
                                            queries[:, 5:6]], dim=1)],
                                dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def embed_query(self, queries, query_structure, idx):
        r"""
        :param query_structure: e.g. `((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))`
        :param queries: Tensor. shape `[batch_size, M]`,
                        where `M` is the number of elements in query_structure
                        (6 in the above examples)
        :param idx: which `column` to start in tensor queries
        """
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                axis_entity_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=queries[:, idx])
                axis_entity_embedding = self.angle_scale(
                    axis_entity_embedding, self.axis_scale)
                axis_entity_embedding = convert_to_axis(
                    axis_entity_embedding)

                if self.use_cuda:
                    arg_entity_embedding = torch.zeros_like(
                        axis_entity_embedding).cuda()
                else:
                    arg_entity_embedding = torch.zeros_like(
                        axis_entity_embedding)
                idx += 1

                axis_embedding = axis_entity_embedding
                arg_embedding = arg_entity_embedding
            else:
                axis_embedding, arg_embedding, idx = self.embed_query(
                    queries, query_structure[0], idx)

            for i in range(len(query_structure[-1])):
                # negation
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    axis_embedding, arg_embedding = self.negation(
                        axis_embedding, arg_embedding)
                # projection (for 'r')
                else:
                    rel_indices = queries[:, idx]
                    axis_embedding, arg_embedding = self.projection(
                        axis_embedding, arg_embedding, rel_indices)
                idx += 1
        else:
            # intersection
            axis_embedding_list = []
            arg_embedding_list = []
            for i in range(len(query_structure)):
                axis_embedding, arg_embedding, idx = self.embed_query(
                    queries, query_structure[i], idx)
                axis_embedding_list.append(axis_embedding)
                arg_embedding_list.append(arg_embedding)

            stacked_axis_embeddings = torch.stack(axis_embedding_list)
            stacked_arg_embeddings = torch.stack(arg_embedding_list)
            axis_embedding, arg_embedding = self.intersection(
                stacked_axis_embeddings, stacked_arg_embeddings)

        return axis_embedding, arg_embedding, idx

    def cal_logit_model(self, entity_axis_embedding, query_axis_embedding, query_arg_embedding):
        entity_axis_embedding = self.angle_scale(
            entity_axis_embedding, self.axis_scale)
        entity_axis_embedding = convert_to_axis(entity_axis_embedding)

        low_query = query_axis_embedding - query_arg_embedding
        up_query = query_axis_embedding + query_arg_embedding

        # inside distance
        if self.distance_type == "cosine":
            distance2axis = torch.abs(
                1 - torch.cos(entity_axis_embedding - query_axis_embedding))
            distance_base = torch.abs(1 - torch.cos(query_arg_embedding))
        elif self.distance_type == "sine":
            distance2axis = torch.abs(
                torch.sin(entity_axis_embedding - query_axis_embedding))
            distance_base = torch.abs(torch.sin(query_arg_embedding))
        elif self.distance_type == "angle":
            distance2axis = torch.abs(
                entity_axis_embedding - query_axis_embedding)
            distance_base = torch.abs(query_arg_embedding)
        distance_in = torch.min(distance2axis, distance_base)
        indicator_in = distance2axis < distance_base

        # outside distance
        if self.distance_type == "cosine":
            distance_out = torch.min(torch.abs(1 - torch.cos(entity_axis_embedding - low_query)),
                                     torch.abs(1 - torch.cos(entity_axis_embedding - up_query)))
            distance_out[indicator_in] = 0.
        elif self.distance_type == "sine":
            distance_out = torch.min(torch.abs(torch.sin(entity_axis_embedding - low_query)),
                                     torch.abs(torch.sin(entity_axis_embedding - up_query)))
            distance_out[indicator_in] = 0.
        elif self.distance_type == "angle":
            distance_out = torch.min(
                (entity_axis_embedding - low_query), (entity_axis_embedding - up_query))
            distance_out[indicator_in] = 0.

        # axis distance
        distance_axis = torch.abs(entity_axis_embedding - query_axis_embedding)

        distance = (1 - self.psi) * (torch.norm(distance_out, p=1, dim=-1) + self.cen * torch.norm(distance_in, p=1, dim=-1)) + \
            self.psi * torch.norm(distance_axis, p=1, dim=-1)

        logit = self.gamma - distance * self.modulus

        return logit

    def forward_model(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_union_idxs = [], []
        all_axis_embeddings, all_arg_embeddings = [], []
        all_union_axis_embeddings, all_union_arg_embeddings = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                axis_embedding, arg_embedding, _ = \
                    self.embed_query(self.transform_union_query(batch_queries_dict[query_structure], query_structure),
                                     self.transform_union_structure(query_structure), 0)
                all_union_axis_embeddings.append(axis_embedding)
                all_union_arg_embeddings.append(arg_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                axis_embedding, arg_embedding, _ = \
                    self.embed_query(
                        batch_queries_dict[query_structure], query_structure, 0)
                all_axis_embeddings.append(axis_embedding)
                all_arg_embeddings.append(arg_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])
        if len(all_axis_embeddings) > 0:
            all_axis_embeddings = torch.cat(
                all_axis_embeddings, dim=0).unsqueeze(1)
            all_arg_embeddings = torch.cat(
                all_arg_embeddings, dim=0).unsqueeze(1)
        if len(all_union_axis_embeddings) > 0:
            all_union_axis_embeddings = torch.cat(
                all_union_axis_embeddings, dim=0).unsqueeze(1)
            all_union_arg_embeddings = torch.cat(
                all_union_arg_embeddings, dim=0).unsqueeze(1)
            all_union_axis_embeddings = all_union_axis_embeddings.view(
                all_union_axis_embeddings.shape[0] // 2, 2, 1, -1)
            all_union_arg_embeddings = all_union_arg_embeddings.view(
                all_union_arg_embeddings.shape[0] // 2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                # positive samples for non-union queries in this batch
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding_cone = torch.index_select(self.entity_embedding, dim=0,
                                                             index=positive_sample_regular)
                positive_embedding_cone = positive_embedding_cone.unsqueeze(1)
                positive_logit = self.cal_logit_model(
                    positive_embedding_cone, all_axis_embeddings, all_arg_embeddings)

            else:
                positive_logit = torch.Tensor([]).to(
                    self.entity_embedding.device)

            if len(all_union_axis_embeddings) > 0:
                # positive samples for union queries in this batch
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding_cone = torch.index_select(
                    self.entity_embedding, dim=0, index=positive_sample_union)
                positive_embedding_cone = positive_embedding_cone.unsqueeze(
                    1).unsqueeze(1)
                positive_union_logit = self.cal_logit_model(
                    positive_embedding_cone, all_union_axis_embeddings, all_union_arg_embeddings)
                positive_union_logit = torch.max(
                    positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor(
                    []).to(self.entity_embedding.device)
            positive_logit = torch.cat(
                [positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding_cone = torch.index_select(self.entity_embedding, dim=0,
                                                             index=negative_sample_regular.view(-1))
                negative_embedding_cone = negative_embedding_cone.view(
                    batch_size, negative_size, -1)
                negative_logit = self.cal_logit_model(negative_embedding_cone,
                                                      all_axis_embeddings, all_arg_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(
                    self.entity_embedding.device)

            if len(all_union_axis_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding_cone = torch.index_select(self.entity_embedding, dim=0,
                                                             index=negative_sample_union.view(-1))
                negative_embedding_cone = negative_embedding_cone.view(
                    batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_model(negative_embedding_cone,
                                                            all_union_axis_embeddings, all_union_arg_embeddings)
                negative_union_logit = torch.max(
                    negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor(
                    []).to(self.entity_embedding.device)
            negative_logit = torch.cat(
                [negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(
            train_iterator)

        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        # group queries with same structure
        for i, query in enumerate(batch_queries):
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(
                    batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(
                    batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _ = model(
            positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader,
                  query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in \
                    tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, idxs = model(
                    None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                if len(argsort) == args.test_batch_size:
                    # achieve the ranking of all entities
                    ranking = ranking.scatter_(
                        1, argsort, model.batch_entity_range)
                else:  # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1).cuda()
                                                   )  # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1)
                                                   )  # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(
                        easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(
                            num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(
                            num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    # only take indices that belong to the hard answers
                    cur_ranking = cur_ranking[masks]

                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean(
                        (cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' %
                                 (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum(
                    [log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(
                logs[query_structure])

        return metrics
