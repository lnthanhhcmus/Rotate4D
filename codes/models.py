import os
import logging
import math
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader
from data import BatchType, ModeType, TestDataset


class KGEModel(nn.Module, ABC):
    """
    Must define
        `self.entity_embedding`
        `self.relation_embedding`
    in the subclasses.
    """

    @abstractmethod
    def func(self, head, rel, tail, batch_type):
        """
        Different tensor shape for different batch types.
        BatchType.SINGLE:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.HEAD_BATCH:
            head: [batch_size, negative_sample_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.TAIL_BATCH:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, negative_sample_size, hidden_dim]
        """
        ...

    def forward(self, sample, batch_type=BatchType.SINGLE):
        """
        Given the indexes in `sample`, extract the corresponding embeddings,
        and call func().

        Args:
            batch_type: {SINGLE, HEAD_BATCH, TAIL_BATCH},
                - SINGLE: positive samples in training, and all samples in validation / testing,
                - HEAD_BATCH: (?, r, t) tasks in training,
                - TAIL_BATCH: (h, r, ?) tasks in training.

            sample: different format for different batch types.
                - SINGLE: tensor with shape [batch_size, 3]
                - {HEAD_BATCH, TAIL_BATCH}: (positive_sample, negative_sample)
                    - positive_sample: tensor with shape [batch_size, 3]
                    - negative_sample: tensor with shape [batch_size, negative_sample_size]
        """
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
        '''
        A single train step. Apply back-propation and return the loss
        '''

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
    def test_step(model, data_reader, mode, args, rel_type):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.HEAD_BATCH,
                rel_type
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.TAIL_BATCH,
                rel_type
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []
        logs_rel = defaultdict(list)  # logs for every relation
        logs_h = defaultdict(list) # logs for head batch
        logs_t = defaultdict(list) # logs for tail batch

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, batch_type, relation_type in test_dataset:
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

                        if batch_type == BatchType.HEAD_BATCH:
                            logs_h[relation_type[i].item()].append(log)
                        else:
                            logs_t[relation_type[i].item()].append(log)

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

        metrics_h = defaultdict(dict)
        for rel in logs_h:
            for metric in logs_h[rel][0].keys():
                metrics_h[rel][metric] = sum([log[metric] for log in logs_h[rel]]) / len(logs_h[rel])

        metrics_t = defaultdict(dict)
        for rel in logs_t:
            for metric in logs_t[rel][0].keys():
                metrics_t[rel][metric] = sum([log[metric] for log in logs_t[rel]]) / len(logs_t[rel])

        return metrics, metrics_rel, metrics_h, metrics_t


class Rotate3D(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, p_norm):
        super().__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.p = p_norm

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 3))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 4))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # Initialize bias to 1
        nn.init.ones_(
            tensor=self.relation_embedding[:, 3*hidden_dim:4*hidden_dim]
        )

        self.pi = 3.14159262358979323846

    def func(self, head, rel, tail, batch_type):
        head_i, head_j, head_k = torch.chunk(head, 3, dim=2)
        beta_1, beta_2, theta, bias = torch.chunk(rel, 4, dim=2)
        tail_i, tail_j, tail_k = torch.chunk(tail, 3, dim=2)

        bias = torch.abs(bias)

        # Make phases of relations uniformly distributed in [-pi, pi]
        beta_1 = beta_1 / (self.embedding_range.item() / self.pi)
        beta_2 = beta_2 / (self.embedding_range.item() / self.pi)
        theta = theta / (self.embedding_range.item() / self.pi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Obtain representation of the rotation axis
        rel_i = torch.cos(beta_1)
        rel_j = torch.sin(beta_1)*torch.cos(beta_2)
        rel_k = torch.sin(beta_1)*torch.sin(beta_2)

        C = rel_i*head_i + rel_j*head_j + rel_k*head_k
        C = C*(1-cos_theta)

        # Rotate the head entity
        new_head_i = head_i*cos_theta + C*rel_i + sin_theta*(rel_j*head_k-head_j*rel_k)
        new_head_j = head_j*cos_theta + C*rel_j - sin_theta*(rel_i*head_k-head_i*rel_k)
        new_head_k = head_k*cos_theta + C*rel_k + sin_theta*(rel_i*head_j-head_i*rel_j)

        score_i = new_head_i*bias - tail_i
        score_j = new_head_j*bias - tail_j
        score_k = new_head_k*bias - tail_k

        score = torch.stack([score_i, score_j, score_k], dim=0)
        score = score.norm(dim=0, p=self.p)
        score = self.gamma.item() - score.sum(dim=2)
        return score


class RotatE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, p_norm):
        super().__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.p = p_norm

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.pi = 3.14159262358979323846

    def func(self, head, rel, tail, batch_type):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = rel/(self.embedding_range.item()/self.pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation - re_tail
        im_score = re_head * im_relation + im_head * re_relation - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0, p=self.p)
        score = self.gamma.item() - score.sum(dim=2)
        return score


class Rotate4D_v1(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, p_norm, dataset):
        super().__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.p = p_norm

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 4))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 4))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if (dataset == 'wn18rr'):
            nn.init.uniform_(tensor=self.relation_embedding[:, 3*hidden_dim:4*hidden_dim], a = 1, b = 2)
        elif (dataset == 'FB15k-237'):
            nn.init.uniform_(tensor=self.relation_embedding[:, 3*hidden_dim:4*hidden_dim], a = 0, b = 1)
        else:
            nn.init.ones_(tensor=self.relation_embedding[:, 3*hidden_dim:4*hidden_dim])

        self.pi = 3.14159262358979323846

    def func(self, head, rel, tail, batch_type):
        u_h, x_h, y_h, z_h = torch.chunk(head, 4, dim = 2)
        alpha_1, alpha_2, alpha_3, bias = torch.chunk(rel, 4, dim = 2)
        u_t, x_t, y_t, z_t = torch.chunk(tail, 4, dim = 2)

        bias = torch.abs(bias)

        # Make phases of relations uniformly distributed in [-pi, pi]
        alpha_1 = alpha_1 / (self.embedding_range.item() / self.pi)
        alpha_2 = alpha_2 / (self.embedding_range.item() / self.pi)
        alpha_3 = alpha_3 / (self.embedding_range.item() / self.pi)

        # Obtain representation of the rotation axis
        a_r = torch.cos(alpha_1)
        b_r = torch.sin(alpha_1)*torch.cos(alpha_2)
        c_r = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.cos(alpha_3)
        d_r = torch.sin(alpha_1)*torch.sin(alpha_2)*torch.sin(alpha_3)

        score_u = (a_r*u_h - b_r*x_h - c_r*y_h - d_r*z_h)*bias - u_t
        score_x = (a_r*x_h + b_r*u_h + c_r*z_h - d_r*y_h)*bias - x_t
        score_y = (a_r*y_h - b_r*z_h + c_r*u_h + d_r*x_h)*bias - y_t
        score_z = (a_r*z_h + b_r*y_h - c_r*x_h + d_r*u_h)*bias - z_t
        
        score = torch.stack([score_u, score_x, score_y, score_z], dim=0)
        score = score.norm(dim=0, p=self.p)
        score = self.gamma.item() - score.sum(dim=2)
        return score


class Rotate4D_v2(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, p_norm, dataset):
        super().__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.p = p_norm

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 4))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 7))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if (dataset == 'wn18rr'):
            nn.init.uniform_(tensor=self.relation_embedding[:, 6*hidden_dim:7*hidden_dim], a = 1, b = 2)
        elif (dataset == 'FB15k-237'):
            nn.init.uniform_(tensor=self.relation_embedding[:, 6*hidden_dim:7*hidden_dim], a = 0, b = 1)
        else:
            nn.init.ones_(tensor=self.relation_embedding[:, 6*hidden_dim:7*hidden_dim])

        self.pi = 3.14159262358979323846

    def func(self, head, rel, tail, batch_type):
        a_h, b_h, c_h, d_h = torch.chunk(head, 4, dim = 2)
        alpha_1, alpha_2, beta_1, beta_2, theta, gamma, bias = torch.chunk(rel, 7, dim = 2)
        a_t, b_t, c_t, d_t = torch.chunk(tail, 4, dim = 2)

        bias = torch.abs(bias)

        # Make phases of relations uniformly distributed in [-pi, pi]
        alpha_1 = alpha_1 / (self.embedding_range.item() / self.pi)
        alpha_2 = alpha_2 / (self.embedding_range.item() / self.pi)
        beta_1 = beta_1 / (self.embedding_range.item() / self.pi)
        beta_2 = beta_2 / (self.embedding_range.item() / self.pi)
        theta = theta / (self.embedding_range.item() / self.pi)
        gamma = gamma / (self.embedding_range.item() / self.pi)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)

        # Obtain representation of the rotation axis
        x_p = torch.cos(alpha_1)
        y_p = torch.sin(alpha_1)*torch.cos(alpha_2)
        z_p = torch.sin(alpha_1)*torch.sin(alpha_2)

        x_q = torch.cos(beta_1)
        y_q = torch.sin(beta_1)*torch.cos(beta_2)
        z_q = torch.sin(beta_1)*torch.sin(beta_2)

        A = cos_theta*a_h - sin_theta*(x_p*b_h + y_p*c_h + z_p*d_h)
        B = cos_theta*b_h + sin_theta*(x_p*a_h + y_p*d_h - z_p*c_h)
        C = cos_theta*c_h + sin_theta*(y_p*a_h - x_p*d_h + z_p*b_h)
        D = cos_theta*d_h + sin_theta*(z_p*a_h + x_p*c_h - y_p*b_h)

        score_a = (cos_gamma*A - sin_gamma*(B*x_q + C*y_q + D*z_q))*bias - a_t
        score_b = (cos_gamma*B + sin_gamma*(A*x_q + C*z_q - D*y_q))*bias - b_t
        score_c = (cos_gamma*C + sin_gamma*(A*y_q - B*z_q + D*x_q))*bias - c_t
        score_d = (cos_gamma*D + sin_gamma*(A*z_q + B*y_q - C*x_q))*bias - d_t
        
        score = torch.stack([score_a, score_b, score_c, score_d], dim=0)
        score = score.norm(dim=0, p=self.p)
        score = self.gamma.item() - score.sum(dim=2)
        return score