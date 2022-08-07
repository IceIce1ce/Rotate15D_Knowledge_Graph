import os
import logging
import math
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import BatchType, ModeType, TestDataset

class KGEModel(nn.Module, ABC):
    @abstractmethod
    def func(self, head, rel, tail, batch_type):
        ...

    def forward(self, sample, batch_type=BatchType.SINGLE):
        if batch_type == BatchType.SINGLE:
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
        elif batch_type == BatchType.HEAD_BATCH:
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
        elif batch_type == BatchType.TAIL_BATCH:
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        else:
            raise ValueError('batch_type %s not supported!'.format(batch_type))

        return self.func(head, relation, tail, batch_type), (head, tail)

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()
        # negative scores
        negative_score, _ = model((positive_sample, negative_sample), batch_type=batch_type)
        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)
        # positive scores
        positive_score, ent = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        loss = (positive_sample_loss + negative_sample_loss) / 2
        if args.regularization:
            # Use regularization
            regularization = args.regularization * (
                ent[0].norm(p=2)**2 +
                ent[1].norm(p=2)**2
            ) / ent[0].shape[0]
            loss = loss + regularization
        else:
            regularization = torch.tensor([0])
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
            'regularization': regularization.item()
        }
        return log

    @staticmethod
    def test_step(model, data_reader, mode, args):
        model.eval()
        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.HEAD_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )
        test_dataloader_tail = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.TAIL_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        logs = []
        logs_rel = defaultdict(list)  # logs for every relation
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, batch_type in test_dataset:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()
                    batch_size = positive_sample.size(0)
                    score, _ = model((positive_sample, negative_sample), batch_type)
                    score += filter_bias
                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)
                    if batch_type == BatchType.HEAD_BATCH:
                        positive_arg = positive_sample[:, 0]
                    elif batch_type == BatchType.TAIL_BATCH:
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)
                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1
                        rel = positive_sample[i][1].item()
                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        log = {
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        }
                        logs.append(log)
                        logs_rel[rel].append(log)

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... ({}/{})'.format(step, total_steps))
                    step += 1
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        metrics_rel = defaultdict(dict)
        for rel in logs_rel:
            for metric in logs_rel[rel][0].keys():
                metrics_rel[rel][metric] = sum([log[metric] for log in logs_rel[rel]]) / len(logs_rel[rel])

        return metrics, metrics_rel

class Rotate16D(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, p_norm):
        super().__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.p = p_norm
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), requires_grad=False)
        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 15))
        nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 16))
        nn.init.uniform_(tensor=self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.ones_(tensor=self.relation_embedding[:, 15*hidden_dim:16*hidden_dim])
        self.pi = 3.14159262358979323846

    def func(self, head, rel, tail, batch_type):
        head_e1, head_e2, head_e3, head_e4, head_e5, head_e6, head_e7, head_e8, head_e9, \
            head_e10, head_e11, head_e12, head_e13, head_e14, head_e15 = torch.chunk(head, 15, dim=2)
        alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_7, alpha_8, alpha_9, \
            alpha_10, alpha_11, alpha_12, alpha_13, alpha_14, theta, bias = torch.chunk(rel, 16, dim=2)
        tail_e1, tail_e2, tail_e3, tail_e4, tail_e5, tail_e6, tail_e7, tail_e8, tail_e9, \
            tail_e10, tail_e11, tail_e12, tail_e13, tail_e14, tail_e15 = torch.chunk(tail, 15, dim=2)
        bias = torch.abs(bias)
        # Make phases of relations uniformly distributed in [-pi, pi]
        alpha_1 = alpha_1 / (self.embedding_range.item() / self.pi)
        alpha_2 = alpha_2 / (self.embedding_range.item() / self.pi)
        alpha_3 = alpha_3 / (self.embedding_range.item() / self.pi)
        alpha_4 = alpha_4 / (self.embedding_range.item() / self.pi)
        alpha_5 = alpha_5 / (self.embedding_range.item() / self.pi)
        alpha_6 = alpha_6 / (self.embedding_range.item() / self.pi)
        alpha_7 = alpha_7 / (self.embedding_range.item() / self.pi)
        alpha_8 = alpha_8 / (self.embedding_range.item() / self.pi)
        alpha_9 = alpha_9 / (self.embedding_range.item() / self.pi)
        alpha_10 = alpha_10 / (self.embedding_range.item() / self.pi)
        alpha_11 = alpha_11 / (self.embedding_range.item() / self.pi)
        alpha_12 = alpha_12 / (self.embedding_range.item() / self.pi)
        alpha_13 = alpha_13 / (self.embedding_range.item() / self.pi)
        alpha_14 = alpha_14 / (self.embedding_range.item() / self.pi)
        theta = theta / (self.embedding_range.item() / self.pi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        # Obtain representation of the rotation axis
        rel_e1 = torch.cos(alpha_1)
        rel_e2 = torch.sin(alpha_1)*torch.cos(alpha_2)
        rel_e3 = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.cos(alpha_3)
        rel_e4 = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.sin(alpha_3)*torch.cos(alpha_4)
        rel_e5 = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.sin(alpha_3)*torch.sin(alpha_4)*torch.cos(alpha_5)
        rel_e6 = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.sin(alpha_3)*torch.sin(alpha_4)*torch.sin(alpha_5)*torch.cos(alpha_6)
        rel_e7 = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.sin(alpha_3)*torch.sin(alpha_4)*torch.sin(alpha_5)*torch.sin(alpha_6)*torch.cos(alpha_7)
        rel_e8 = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.sin(alpha_3)*torch.sin(alpha_4)*torch.sin(alpha_5)*torch.sin(alpha_6)*torch.sin(alpha_7)* \
                 torch.cos(alpha_8)
        rel_e9 = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.sin(alpha_3)*torch.sin(alpha_4)*torch.sin(alpha_5)*torch.sin(alpha_6)*torch.sin(alpha_7)* \
                 torch.sin(alpha_8)*torch.cos(alpha_9)
        rel_e10 = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.sin(alpha_3)*torch.sin(alpha_4)*torch.sin(alpha_5)*torch.sin(alpha_6)*torch.sin(alpha_7)* \
                 torch.sin(alpha_8)*torch.sin(alpha_9)*torch.cos(alpha_10)
        rel_e11 = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.sin(alpha_3)*torch.sin(alpha_4)*torch.sin(alpha_5)*torch.sin(alpha_6)*torch.sin(alpha_7)* \
                 torch.sin(alpha_8)*torch.sin(alpha_9)*torch.sin(alpha_10)*torch.cos(alpha_11)
        rel_e12 = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.sin(alpha_3)*torch.sin(alpha_4)*torch.sin(alpha_5)*torch.sin(alpha_6)*torch.sin(alpha_7)* \
                 torch.sin(alpha_8)*torch.sin(alpha_9)*torch.sin(alpha_10)*torch.sin(alpha_11)*torch.cos(alpha_12)
        rel_e13 = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.sin(alpha_3)*torch.sin(alpha_4)*torch.sin(alpha_5)*torch.sin(alpha_6)*torch.sin(alpha_7)* \
                 torch.sin(alpha_8)*torch.sin(alpha_9)*torch.sin(alpha_10)*torch.sin(alpha_11)*torch.sin(alpha_12)*torch.cos(alpha_13)
        rel_e14 = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.sin(alpha_3)*torch.sin(alpha_4)*torch.sin(alpha_5)*torch.sin(alpha_6)*torch.sin(alpha_7)* \
                 torch.sin(alpha_8)*torch.sin(alpha_9)*torch.sin(alpha_10)*torch.sin(alpha_11)*torch.sin(alpha_12)*torch.sin(alpha_13)*torch.cos(alpha_14)
        rel_e15 = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.sin(alpha_3)*torch.sin(alpha_4)*torch.sin(alpha_5)*torch.sin(alpha_6)*torch.sin(alpha_7)* \
                 torch.sin(alpha_8)*torch.sin(alpha_9)*torch.sin(alpha_10)*torch.sin(alpha_11)*torch.sin(alpha_12)*torch.sin(alpha_13)*torch.sin(alpha_14)
        C = rel_e1*head_e1 + rel_e2*head_e2 + rel_e3*head_e3 + rel_e4*head_e4 + rel_e5*head_e5 + rel_e6*head_e6 + rel_e7*head_e7 + \
            rel_e8*head_e8 + rel_e9*head_e9 + rel_e10*head_e10 + rel_e11*head_e11 + rel_e12*head_e12 + rel_e13*head_e13 + \
            rel_e14*head_e14 + rel_e15*head_e15
        C = C*(1-cos_theta)
        # Rotate the head entity
        new_head_e1 = head_e1*cos_theta + C*rel_e1 + sin_theta*(rel_e2*head_e3 - rel_e3*head_e2 + rel_e4*head_e5 - rel_e5*head_e4 -
                                                                rel_e6*head_e7 + rel_e7*head_e6 + rel_e8*head_e9 - rel_e9*head_e8 -
                                                                rel_e10*head_e11 + rel_e11*head_e10 - rel_e12*head_e13 +
                                                                rel_e13*head_e12 + rel_e14*head_e15 - rel_e15*head_e14)
        new_head_e2 = head_e2*cos_theta + C*rel_e2 + sin_theta*(-rel_e1*head_e3 + rel_e3*head_e1 + rel_e4*head_e6 - rel_e6*head_e4 +
                                                                rel_e5*head_e7 - rel_e7*head_e5 + rel_e8*head_e10 - rel_e10*head_e8 +
                                                                rel_e9*head_e11 - rel_e11*head_e9 - rel_e12*head_e14 +
                                                                rel_e14*head_e12 - rel_e13*head_e15 + rel_e15*head_e13)
        new_head_e3 = head_e3*cos_theta + C*rel_e3 + sin_theta*(rel_e1*head_e2 - rel_e2*head_e1 + rel_e4*head_e7 - rel_e7*head_e4 -
                                                                rel_e5*head_e6 + rel_e6*head_e5 + rel_e8*head_e11 - rel_e11*head_e8 -
                                                                rel_e9*head_e10 + rel_e10*head_e9 - rel_e12*head_e15 +
                                                                rel_e15*head_e12 + rel_e13*head_e14 - rel_e14*head_e13)
        new_head_e4 = head_e4*cos_theta + C*rel_e4 + sin_theta*(-rel_e1*head_e5 + rel_e5*head_e1 - rel_e2*head_e6 + rel_e6*head_e2 -
                                                                rel_e3*head_e7 + rel_e7*head_e3 + rel_e8*head_e12 - rel_e12*head_e8 +
                                                                rel_e9*head_e13 - rel_e13*head_e9 + rel_e10*head_e14 - 
                                                                rel_e14*head_e10 + rel_e11*head_e15 - rel_e15*head_e11)
        new_head_e5 = head_e5*cos_theta + C*rel_e5 + sin_theta*(rel_e1*head_e4 - rel_e4*head_e1 - rel_e2*head_e7 + rel_e7*head_e2 +
                                                                rel_e3*head_e6 - rel_e6*head_e3 + rel_e8*head_e13 - rel_e13*head_e8 -
                                                                rel_e9*head_e12 + rel_e12*head_e9 + rel_e10*head_e15 -
                                                                rel_e15*head_e10 - rel_e11*head_e14 + rel_e14*head_e11)
        new_head_e6 = head_e6*cos_theta + C*rel_e6 + sin_theta*(rel_e1*head_e7 - rel_e7*head_e1 + rel_e2*head_e4 - rel_e4*head_e2 -
                                                                rel_e3*head_e5 + rel_e5*head_e3 + rel_e8*head_e14 - rel_e14*head_e8 -
                                                                rel_e9*head_e15 + rel_e15*head_e9 - rel_e10*head_e12 + 
                                                                rel_e12*head_e10 + rel_e11*head_e13 - rel_e13*head_e11)
        new_head_e7 = head_e7*cos_theta + C*rel_e7 + sin_theta*(-rel_e1*head_e6 + rel_e6*head_e1 + rel_e2*head_e5 - rel_e5*head_e2 +
                                                                rel_e3*head_e4 - rel_e4*head_e3 + rel_e8*head_e15 - rel_e15*head_e8 +
                                                                rel_e9*head_e14 - rel_e14*head_e9 - rel_e10*head_e13 +
                                                                rel_e13*head_e10 - rel_e11*head_e12 + rel_e12*head_e11)
        new_head_e8 = head_e8*cos_theta + C*rel_e8 + sin_theta*(-rel_e1*head_e9 + rel_e9*head_e1 - rel_e2*head_e10 + rel_e10*head_e2 -
                                                                rel_e3*head_e11 + rel_e11*head_e3 - rel_e4*head_e12 + rel_e12*head_e4 -
                                                                rel_e5*head_e13 + rel_e13*head_e5 - rel_e6*head_e14 +
                                                                rel_e14*head_e6 - rel_e7*head_e15 + rel_e15*head_e7)
        new_head_e9 = head_e9*cos_theta + C*rel_e9 + sin_theta*(rel_e1*head_e8 - rel_e8*head_e1 - rel_e2*head_e11 + rel_e11*head_e2 +
                                                                rel_e3*head_e10 - rel_e10*head_e3 - rel_e4*head_e13 + rel_e13*head_e4 +
                                                                rel_e5*head_e12 - rel_e12*head_e5 + rel_e6*head_e15 -
                                                                rel_e15*head_e6 - rel_e7*head_e14 + rel_e14*head_e7)
        new_head_e10 = head_e10*cos_theta + C*rel_e10 + sin_theta*(rel_e1*head_e11 - rel_e11*head_e1 + rel_e2*head_e8 - rel_e8*head_e2 -
                                                                   rel_e3*head_e9 + rel_e9*head_e3 - rel_e4*head_e14 + rel_e14*head_e4 -
                                                                   rel_e5*head_e15 + rel_e15*head_e5 + rel_e6*head_e12 - 
                                                                   rel_e12*head_e6 + rel_e7*head_e13 - rel_e13*head_e7)
        new_head_e11 = head_e11*cos_theta + C*rel_e11 + sin_theta*(-rel_e1*head_e10 + rel_e10*head_e1 + rel_e2*head_e9 - rel_e9*head_e2 +
                                                                   rel_e3*head_e8 - rel_e8*head_e3 - rel_e4*head_e15 + rel_e15*head_e4 +
                                                                   rel_e5*head_e14 - rel_e14*head_e5 - rel_e6*head_e13 + 
                                                                   rel_e13*head_e6 + rel_e7*head_e12 - rel_e12*head_e7)
        new_head_e12 = head_e12*cos_theta + C*rel_e12 + sin_theta*(rel_e1*head_e13 - rel_e13*head_e1 + rel_e2*head_e14 - rel_e14*head_e2 +
                                                                   rel_e3*head_e15 - rel_e15*head_e3 + rel_e4*head_e8 - rel_e8*head_e4 -
                                                                   rel_e5*head_e9 + rel_e9*head_e5 - rel_e6*head_e10 + 
                                                                   rel_e10*head_e6 - rel_e7*head_e11 + rel_e11*head_e7)
        new_head_e13 = head_e13*cos_theta + C*rel_e13 + sin_theta*(-rel_e1*head_e12 + rel_e12*head_e1 + rel_e2*head_e15 - rel_e15*head_e2 -
                                                                   rel_e3*head_e14 + rel_e14*head_e3 + rel_e4*head_e9 - rel_e9*head_e4 +
                                                                   rel_e5*head_e8 - rel_e8*head_e5 + rel_e6*head_e11 - 
                                                                   rel_e11*head_e6 - rel_e7*head_e10 + rel_e10*head_e7)
        new_head_e14 = head_e14*cos_theta + C*rel_e14 + sin_theta*(-rel_e1*head_e15 + rel_e15*head_e1 - rel_e2*head_e12 + rel_e12*head_e2 +
                                                                   rel_e3*head_e13 - rel_e13*head_e3 + rel_e4*head_e10 - rel_e10*head_e4 -
                                                                   rel_e5*head_e11 + rel_e11*head_e5 + rel_e6*head_e8 -
                                                                   rel_e8*head_e6 + rel_e7*head_e9 - rel_e9*head_e7)
        new_head_e15 = head_e15*cos_theta + C*rel_e15 + sin_theta*(rel_e1*head_e14 - rel_e14*head_e1 - rel_e2*head_e13 + rel_e13*head_e2 -
                                                                   rel_e3*head_e12 + rel_e12*head_e3 + rel_e4*head_e11 - rel_e11*head_e4 +
                                                                   rel_e5*head_e10 - rel_e10*head_e5 - rel_e6*head_e9 + 
                                                                   rel_e9*head_e6 + rel_e7*head_e8 - rel_e8*head_e7)
        # Compute the score
        score_e1 = new_head_e1*bias - tail_e1
        score_e2 = new_head_e2*bias - tail_e2
        score_e3 = new_head_e3*bias - tail_e3
        score_e4 = new_head_e4*bias - tail_e4
        score_e5 = new_head_e5*bias - tail_e5
        score_e6 = new_head_e6*bias - tail_e6
        score_e7 = new_head_e7*bias - tail_e7
        score_e8 = new_head_e8*bias - tail_e8
        score_e9 = new_head_e9*bias - tail_e9
        score_e10 = new_head_e10*bias - tail_e10
        score_e11 = new_head_e11*bias - tail_e11
        score_e12 = new_head_e12*bias - tail_e12
        score_e13 = new_head_e13*bias - tail_e13
        score_e14 = new_head_e14*bias - tail_e14
        score_e15 = new_head_e15*bias - tail_e15
        score = torch.stack([score_e1, score_e2, score_e3, score_e4, score_e5, score_e6, score_e7, score_e8, 
                             score_e9, score_e10, score_e11, score_e12, score_e13, score_e14, score_e15], dim=0)
        score = score.norm(dim=0, p=self.p)
        score = self.gamma.item() - score.sum(dim=2)
        return score