from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory

import torch
import torch.utils.data as data_utils


class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)

        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        return dataset


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        print("self.mask_prob", self.mask_prob)
        self.mask_token = mask_token
        print("self.mask_token", self.mask_token)
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = []
        labels = []
        for s in seq: # касательно каждого s заполняем токен и лейбл
            prob = self.rng.random()  # сгенерили для конкретного s вероятность uniform, deterministic
            if prob < self.mask_prob:  # если prob меньше self.mask_prob == 0.15: TLDR С ВЕРОЯТНОСТЬЮ 80% ЗАПОЛНИМ ТОКЕН МАСКТОКЕНОМ, А НЕ ЭТИМ ITEM INDEX
                prob /= self.mask_prob  # то сильно бустим, затем...

                if prob < 0.8:
                    tokens.append(self.mask_token)  # если оч слабо то заполняем токен масктокеном - который max(item !INDEX!) + 1 или self.item_count + 1
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))  # если попал в маленькое окошко - то рандомно между 1 и self.item_count (== self.num_items) НАХУЯ?
                else:
                    tokens.append(s) # emergency(?) вариант, но лэйбл всё равно заполнится ненормальным значением - он заполнится ЭТИМ ITEM INDEX

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)  # !!!ЭТО СТРАННО ВЕДЬ СУЩНОСТЬ S [0, MAX INDEX]!!!

        # в итоге каждому s присвоится либо (s, 0) если не замаскирован, либо (mask_token<под вопросом>, s). окей, для чего? !!!ЭТО СТРАННО ВЕДЬ СУЩНОСТЬ S [0, MAX INDEX]!!!

        # видимо max_len это не длина окна подпоследовательности из всей последовательности (если использование такой подпоследовательности ваще в этой реализации будет как-то фигурировать), а то, раньше чего мы 100% забываем
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        # был 140 - стал 100
        # был 90 - стал 90

        # тупа паддинг, если изначальный seq был меньше max_len
        mask_len = self.max_len - len(tokens)

        # прилепляем нули слева. почему нули???? !!!ЭТО СТРАННО ВЕДЬ СУЩНОСТЬ S [0, MAX INDEX]!!!
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        # в итоге каждому s присвоится либо (s, 0) если не замаскирован, либо (mask_token<под вопросом>, s) + в начале будут прилеплены нули, если не дотягивает до max_len, или иначе обрублено [-max_len:]. окей, для чего? !!!ЭТО СТРАННО ВЕДЬ СУЩНОСТЬ S [0, MAX INDEX]!!!
        # tokens    0   0   0   2969, 1574,   957,    1178,   <3707>, 1658,   <3707>, 1117
        # labels    0   0   0   0     0       0       0       2147    0       3177    0
        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]



class BertEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)

