from datasets import dataset_factory
from .bert import BertDataloader, BertTrainDataset, BertEvalDataset
from .ae import AEDataloader

#
import pandas as pd
import pickle
import random
import torch

from tqdm import trange
from collections import Counter
#

DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    AEDataloader.code(): AEDataloader
}


def read_data(prepared_data_path):
    with open(prepared_data_path, 'rb') as handle:
        dataset = pickle.load(handle)
    return dataset

def dataloader_factory(args):
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


    ###
    popularity = Counter()
    for user in range(len(umap)):
        popularity.update(train_data[user])
        popularity.update(val_data[user])
        popularity.update(test_data[user])
    popular_items = sorted(popularity, key=popularity.get, reverse=True)

    negative_samples = {}
    print('Sampling negative items')
    for user in trange(len(umap)):
        seen = set(train_data[user])
        seen.update(val_data[user])
        seen.update(test_data[user])

        samples = []
        for item in popular_items:
            if len(samples) == args.test_negative_sample_size:
                break
            if item in seen:
                continue
            samples.append(item)

        negative_samples[user] = samples
    ###

    # test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
    #                                                  self.user_count, self.item_count,
    #                                                  args.test_negative_sample_size,
    #                                                  args.test_negative_sampling_seed,
    #                                                  self.save_folder)

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





    # dataloader = BertDataloader(args, data)
    # а вот load_dataset происходит в ините ABC даталоадера. щас пойдем логировать че load_dataset выдает
    # в ините ABC даталоадера после dataset.load_dataset() создаются словари self.train, self.val, self.test, которые являются датафреймами В ВИДЕ СЛОВАРЕЙ {ЮЗЕР <int>: [АЙТЕМЫ <int>]}, ГДЕ ЗНАЧЕНИЯ В ИНДЕКСАХ А НЕ РЕАЛЬНЫХ АЙДИ
    # umap и smap - массивы индекс: айди в реальном датасете
    # к слову, в ините даталоадера никакие даталоадеры реально еще не создаются


    # train, val, test = dataloader.get_pytorch_dataloaders()
    # get_pytorch_dataloaders -> _get_train_loader -> _get_train_dataset -> return BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
    # а что этот BertTrainDataset делает? Что-то сложное бертовое: разбираемся шо он там делает с каждым s in seq: ...
    # наверное BertEvalDataset тоже что-то страшное делает

    # еблан решил сделать ебланский мув, окей, повторим
    args.num_items = len(smap)

    return train_torch_dataloader, val_torch_dataloader, test_torch_dataloader

# def load_dataframe(file_path):
#     df = pd.read_csv(file_path, sep='::', header=None)
#     df.columns = ['uid', 'sid', 'rating', 'timestamp']
#     return df
#
# def make_dataloaders(file_path):
#     df = load_dataframe(file_path)

