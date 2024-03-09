from .bert import BertDataloader, BertTrainDataset, BertEvalDataset
from .ae import AEDataloader

#
import pandas as pd
import pickle
import random
import torch

from tqdm import trange
from collections import Counter

import numpy as np
from numpy.random import choice
#

DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    AEDataloader.code(): AEDataloader
}


def read_data(prepared_data_path):
    with open(prepared_data_path, 'rb') as handle:
        dataset = pickle.load(handle)
    return dataset

def make_dataloaders(args):
    data = read_data(args.prepared_data_path)

    train_data = data['train']
    val_data = data['val']
    test_data = data['test']
    umap = data['umap']
    smap = data['smap']

    train_torch_dataset = BertTrainDataset(
        u2seq=      train_data,
        max_len=    args.bert_max_len,
        mask_prob=  args.bert_mask_prob,
        mask_token= len(smap) + 1,
        num_items=  len(smap),
        rng=        random.Random(args.dataloader_random_seed)
    )
    train_torch_dataloader = torch.utils.data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True
    )


    # пробуем родить вал дс и вал лоадер


    # ###
    # popularity = Counter()
    # for user in range(len(umap)):
    #     popularity.update(train_data[user])
    #     popularity.update(val_data[user])
    #     popularity.update(test_data[user])
    # popular_items = sorted(popularity, key=popularity.get, reverse=True)
    #
    # negative_samples = {}
    # print('Sampling negative items')
    # for user in trange(len(umap)):
    #     seen = set(train_data[user])
    #     seen.update(val_data[user])
    #     seen.update(test_data[user])
    #
    #     samples = []
    #     for item in popular_items:
    #         if len(samples) == args.test_negative_sample_size:
    #             break
    #         if item in seen:
    #             continue
    #         samples.append(item)
    #
    #     negative_samples[user] = samples
    # ###

    ###
    popularity = Counter()
    for user in range(len(umap)):
        popularity.update(train_data[user])
        popularity.update(val_data[user])
        popularity.update(test_data[user])
    item_probabilities = {k: v / sum([x for x in popularity.values()]) for k, v in popularity.items()}

    print("PROB CHECK")
    print(item_probabilities[62])
    print(item_probabilities[731])

    negative_samples = {}
    print('Sampling negative items')
    for user in trange(len(umap)):
        seen = set(train_data[user])
        seen.update(val_data[user])
        seen.update(test_data[user])

        np.random.seed(user)

        negative_sampled_interactions = list(choice(list(item_probabilities.keys()), 800, replace=False, p=list(item_probabilities.values())))
        negative_sampled_interactions = [x for x in negative_sampled_interactions if x not in seen]
        negative_sampled_interactions = negative_sampled_interactions[:100]

        negative_samples[user] = negative_sampled_interactions
    ###
    # todo: взять negative_samples полученные из АЛСных экспериментов для 100% идентичности замера ndcg@10.

    print("CHECK")
    print(negative_samples[0][:9])
    print(negative_samples[1][:9])


    val_torch_dataset = BertEvalDataset(
        u2seq=              train_data,
        u2answer=           val_data,
        max_len=            args.bert_max_len,
        mask_token=         len(smap) + 1,
        negative_samples=   negative_samples
    )

    val_torch_dataloader = torch.utils.data.DataLoader(val_torch_dataset, batch_size=args.val_batch_size,
                                       shuffle=False, pin_memory=True)

    test_torch_dataset = BertEvalDataset(
        u2seq=              train_data,
        u2answer=           test_data,
        max_len=            args.bert_max_len,
        mask_token=         len(smap) + 1,
        negative_samples=   negative_samples
    )

    test_torch_dataloader = torch.utils.data.DataLoader(test_torch_dataset, batch_size=args.test_batch_size,
                                                       shuffle=False, pin_memory=True)

    # еблан решил сделать ебланский мув, окей, повторим
    args.num_items = len(smap)

    # на старте каждой эпохи содержимое батчей, выплевываемых train_torch_dataloader, собирается рандомно по-новому. Но negative_samples рандомно сгенерированы один раз до процесса обучения и не меняются по ходу эпох.
    # при этом в трейне для последовательности конкретного юзера конечно будут генериться разные маски на разных эпохах.
    return train_torch_dataloader, val_torch_dataloader, test_torch_dataloader
