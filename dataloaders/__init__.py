from datasets import dataset_factory
from .bert import BertDataloader
from .ae import AEDataloader

#
import pandas as pd
import pickle
#

DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    AEDataloader.code(): AEDataloader
}


def read_data():
    with open('./datasets/dataset_for_bert4rec.pickle', 'rb') as handle:
        dataset = pickle.load(handle)
    return dataset

def dataloader_factory(args):
    # dataset = dataset_factory(args)  # это инстанс класса ML1MDataset, у него есть self.load_ratings_df(), а у родит абстр класса есть def preprocess(self):, а сам этот инстанс вызвается в даталоадере через load_dataset (который дергает preprocess)
    data = read_data()

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

