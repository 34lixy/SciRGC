import numpy as np
import threading
import torch


class BFIndexIPGPU:
    def __init__(self, embeddings, vector_dim, gpu_list=[]):
        self.gpu_list = gpu_list
        assert len(gpu_list) > 0

        self.embedding_list = []
        self.offset_list = []
        self.total_num_embeddings = embeddings.shape[0]
        batch_size = int(torch.ceil(torch.tensor(embeddings.shape[0] / len(gpu_list))))
        self.batch_size = batch_size
        for i in range(len(gpu_list)):
            subset_embeddings = embeddings[batch_size * i: min(batch_size * (i + 1), embeddings.shape[0])]
            self.embedding_list.append(subset_embeddings)
            self.offset_list.append(i * batch_size)

    def dp(self, embeddings, query_embedding, device_id):
        embeddings_tensor = torch.tensor(embeddings, device=device_id).clone().detach()
        query_embedding_tensor = torch.tensor(query_embedding[0], device=device_id).clone().detach()
        product = torch.matmul(embeddings_tensor, query_embedding_tensor)
        return product.unsqueeze(0)


    def gpu_ranking_kernel(self, query_embedding, embeddings, n, device_id, indices_range=None):
        if indices_range is None:
            distances = self.dp(embeddings, query_embedding, device_id)
            I = torch.argsort(-distances, dim=-1)[:, :n]
            D = distances[torch.arange(I.shape[0])[:, None].repeat(1, I.shape[1]), I]
        else:
            if len(indices_range) > 0:
                distances = self.dp(embeddings[indices_range], query_embedding, device_id)
                I = torch.argsort(-distances, dim=-1)[:, :n]
                D = distances[torch.arange(I.shape[0])[:, None].repeat(1, I.shape[1]), I]
                I = indices_range[I]
            else:
                I = torch.tensor([]).reshape(query_embedding.shape[0], 0)
                D = torch.tensor([]).reshape(query_embedding.shape[0], 0)
        return I, D

    def search(self, query_embedding, n, indices_range=None):
        assert len(query_embedding.shape) == 2 and query_embedding.shape[0] == 1

        query_embedding_list = []

        query_embedding = torch.tensor(query_embedding)

        I_list = []
        D_list = []

        for i in range(len(self.gpu_list)):
            query_embedding_list.append(query_embedding)

        if indices_range is None:
            indices_range_list = [None] * len(self.gpu_list)
        else:
            indices_range_list = []

            indices_range_gpu = torch.tensor(indices_range)

            for i in range(len(self.gpu_list)):
                indices_range = indices_range_gpu
                sub_indices_pos = torch.logical_and(indices_range >= i * self.batch_size,
                                                    indices_range < min((i + 1) * self.batch_size,
                                                                        self.total_num_embeddings))
                indices_range_list.append(indices_range[sub_indices_pos] - i * self.batch_size)

        for i in range(len(self.gpu_list)):
            I, D = self.gpu_ranking_kernel(query_embedding_list[i], self.embedding_list[i], n, self.gpu_list[i],
                                           indices_range_list[i])
            I_list.append(I + self.offset_list[i])
            D_list.append(D)

        if len(self.gpu_list) == 1:
            I = I_list[0].cpu().numpy()
            D = D_list[0].cpu().numpy()
        else:
            concated_I = torch.cat([I for I in I_list], dim=1)
            concated_D = torch.cat([D for D in D_list], dim=1)
            I = torch.argsort(-concated_D, dim=-1)[:, :n]
            row_indices = torch.arange(I.shape[0])[:, None].repeat(1, I.shape[1])
            D = concated_D[row_indices, I]
            I = concated_I[row_indices, I]

            I = I.cpu().numpy()
            D = D.cpu().numpy()
        return D, I


class BFIndexIPCPU:

    def __init__(self, embeddings, vector_dim, num_shards=1):
        self.total_num_embeddings = embeddings.shape[0]
        shard_size = int(np.ceil(embeddings.shape[0] / num_shards))
        self.shard_size = shard_size
        self.embedding_shards = []
        self.num_shards = num_shards
        for i in range(0, embeddings.shape[0], shard_size):
            self.embedding_shards.append(embeddings[i: i + shard_size])

    def cpu_ranking_kernel(self, query_embedding, n, shard_number, results, indices_range=None):
        if indices_range is None:
            distances = np.matmul(query_embedding, self.embedding_shards[shard_number].T)
            I = np.argpartition(-distances, n - 1, axis=-1)[:, :n]
            D = distances[np.arange(I.shape[0])[:, np.newaxis].repeat(I.shape[1], axis=1), I]
        else:

            sub_indices_pos = np.logical_and(indices_range >= shard_number * self.shard_size,
                                             indices_range < min((shard_number + 1) * self.shard_size,
                                                                 self.total_num_embeddings))
            indices_range = indices_range[sub_indices_pos] - shard_number * self.shard_size
            if len(indices_range) > 0:
                distances = np.matmul(query_embedding, self.embedding_shards[shard_number][indices_range].T)
                if len(indices_range) <= n:
                    I = np.tile(indices_range, query_embedding.shape[0]).reshape(query_embedding.shape[0],
                                                                                 len(indices_range))
                    D = distances
                else:
                    I = np.argpartition(-distances, n - 1, axis=-1)[:, :n]
                    D = distances[np.arange(I.shape[0])[:, np.newaxis].repeat(I.shape[1], axis=1), I]
                    I = indices_range[I]
            else:
                I = np.array([]).reshape(query_embedding.shape[0], 0)
                D = np.array([]).reshape(query_embedding.shape[0], 0)

        results[shard_number] = (I + shard_number * self.shard_size, D)

    def search(self, query_embedding, n, indices_range=None, requires_precision_conversion=None):

        if indices_range is not None:
            indices_range = np.array(indices_range)
        results = [None] * self.num_shards
        threads = []
        for shard_number in range(self.num_shards):
            t = threading.Thread(target=self.cpu_ranking_kernel,
                                 args=(query_embedding, n, shard_number, results, indices_range))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        I, D = list(zip(*results))
        I = np.concatenate(I, axis=1)
        D = np.concatenate(D, axis=1)

        new_I = np.argsort(-D, axis=-1)[:, :n]
        final_D = D[np.arange(new_I.shape[0])[:, np.newaxis].repeat(new_I.shape[1], axis=1), new_I]
        final_I = I[np.arange(new_I.shape[0])[:, np.newaxis].repeat(new_I.shape[1], axis=1), new_I]
        return final_D, final_I.astype(np.int64)


class BFIndexIP:
    def __init__(self, embeddings, vector_dim, gpu_list=[], num_shards=1):
        if len(gpu_list) == 0:
            self.index = BFIndexIPCPU(embeddings, vector_dim, num_shards)
        else:
            self.index = BFIndexIPGPU(embeddings, vector_dim, gpu_list)

    def search(self, query_embedding, n, indices_range=None, requires_precision_conversion=True):
        return self.index.search(query_embedding, n, indices_range)
