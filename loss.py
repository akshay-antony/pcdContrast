import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class XSampleContrastiveLoss:
    def __init__(
        self, labels_temperature=1, preds_temperature=1, device=torch.device("cuda")
    ):
        self.labels_temperature = labels_temperature
        self.preds_temperature = preds_temperature
        self.device = device

    def __call__(
        self, predicted_embeddings: torch.Tensor, pretrained_embeddings: torch.Tensor
    ):
        # predicted_embeddings: (batch_size, embedding_dim_1)
        # pretrained_embeddings: (batch_size, embedding_dim_2)
        # NB:= check for normalization of embeddings
        predicted_embeddings = F.normalize(predicted_embeddings, dim=1)
        pretrained_embeddings = F.normalize(pretrained_embeddings, dim=1)
        pretrained_soft_labels = (
            pretrained_embeddings @ pretrained_embeddings.T
        )  # B * B
        mask = torch.eye(predicted_embeddings.shape[0], dtype=torch.bool).to(
            device=self.device
        )
        pretrained_soft_labels = pretrained_soft_labels[~mask].view(
            pretrained_soft_labels.shape[0], -1
        )  # B * B-1


        pretrained_soft_labels /= self.labels_temperature
        normalized_pretrained_soft_labels = torch.softmax(
            pretrained_soft_labels, dim=1
        )  # B * B-1
        predicted_similarity_matrix = (
            predicted_embeddings @ predicted_embeddings.T
        )  # B*B
        predicted_similarity_matrix = predicted_similarity_matrix[~mask].view(
            predicted_similarity_matrix.shape[0], -1
        )  # B * B-1
        predicted_similarity_matrix /= self.preds_temperature
        predicted_similarity_matrix_log_softmax = F.log_softmax(
            predicted_similarity_matrix, dim=1
        )
        # print(f"norm, {normalized_pretrained_soft_labels.shape}")
        # print(f"preds, {predicted_similarity_matrix_log_softmax.shape}")
        batch_total_loss = -torch.sum(
            normalized_pretrained_soft_labels * predicted_similarity_matrix_log_softmax,
            dim=1,
        )
        # print(f"after sum: {batch_total_loss.shape}")
        batch_loss = torch.mean(batch_total_loss)
        return batch_loss
