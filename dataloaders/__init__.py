from datasets import dataset_factory
from .bert import BertDataloader, BertTrainDataset, BertEvalDataset
from .ae import AEDataloader

#
import pandas as pd
import pickle
import random
import torch
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





    dataloader = BertDataloader(args, data)
    # а вот load_dataset происходит в ините ABC даталоадера. щас пойдем логировать че load_dataset выдает
    # в ините ABC даталоадера после dataset.load_dataset() создаются словари self.train, self.val, self.test, которые являются датафреймами В ВИДЕ СЛОВАРЕЙ {ЮЗЕР <int>: [АЙТЕМЫ <int>]}, ГДЕ ЗНАЧЕНИЯ В ИНДЕКСАХ А НЕ РЕАЛЬНЫХ АЙДИ
    # umap и smap - массивы индекс: айди в реальном датасете
    # к слову, в ините даталоадера никакие даталоадеры реально еще не создаются


    train, val, test = dataloader.get_pytorch_dataloaders()
    # get_pytorch_dataloaders -> _get_train_loader -> _get_train_dataset -> return BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
    # а что этот BertTrainDataset делает? Что-то сложное бертовое: разбираемся шо он там делает с каждым s in seq: ...
    # наверное BertEvalDataset тоже что-то страшное делает
    return train, val, test

# def load_dataframe(file_path):
#     df = pd.read_csv(file_path, sep='::', header=None)
#     df.columns = ['uid', 'sid', 'rating', 'timestamp']
#     return df
#
# def make_dataloaders(file_path):
#     df = load_dataframe(file_path)

