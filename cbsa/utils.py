import json

import string
import time
import torch
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

def build_dataset(config):

    def load_dataset(path, max_sent_per_example=8):
        contents = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except:
                    print(data)
                sentences = data['sentences']
                labels = data['labels']
                if len(sentences) > max_sent_per_example and max_sent_per_example > 0:
                    for sentence, label in enforce_max_sent_per_example(sentences, labels, max_sent_per_example):
                        token_ids, labels, seq_len, mask = encode_fn(sentence, label)
                        contents.append((token_ids, labels, seq_len, mask))
                else:
                    token_ids, labels, seq_len, mask = encode_fn(sentences, labels)
                    contents.append((token_ids, labels, seq_len, mask))
        return contents

    def enforce_max_sent_per_example(sentences, labels, max_sent_per_example):
        """
        Splits examples with len(sentences) > self.max_sent_per_example into multiple smaller examples
        with len(sentences) <= self.max_sent_per_example.
        Recursively split the list of sentences into two halves until each half
        has len(sentences) < <= self.max_sent_per_example. The goal is to produce splits that are of almost
        equal size to avoid the scenario where all splits are of size
        self.max_sent_per_example then the last split is 1 or 2 sentences
        This will result into losing context around the edges of each examples.
        """
        if len(sentences) > max_sent_per_example and max_sent_per_example > 0:
            i = len(sentences) // 2
            l1 = enforce_max_sent_per_example(sentences[:i], labels[:i], max_sent_per_example)
            l2 = enforce_max_sent_per_example(sentences[i:], labels[i:], max_sent_per_example)
            return l1 + l2
        else:
            return [(sentences, labels)]

    def encode_fn(sentences, labels,  pad_size = 512):
        token = []
        # different models use different seperate token
        if config.model_name == "xlm-roberta-base":
            for s in sentences:
                token = token + config.tokenizer.tokenize(s) + config.tokenizer.tokenize("</s>")
            token = config.tokenizer.tokenize("<s>") + token
        else:
            for s in sentences:
                token = token + config.tokenizer.tokenize(s) + config.tokenizer.tokenize("[SEP]")
            token = config.tokenizer.tokenize("[CLS]") + token

        seq_len = len(token)  # 文本实际长度（填充或截断之前）
        mask = []  # 区分填充部分和非填充部分
        token_ids = config.tokenizer.convert_tokens_to_ids(token)
        if pad_size:
            if len(token_ids) < pad_size:# padding
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token_ids))  # mask
                if config.model_name == "xlm-roberta-base":
                    token_ids += ([1] * (pad_size - len(token_ids)))
                else:
                    token_ids += ([0] * (pad_size - len(token_ids)))
            else:  # no padding
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
                if config.model_name == "xlm-roberta-base":
                    sents_num = token_ids.count(2)
                else:
                    sents_num = token_ids.count(102)
                labels = labels[:sents_num]
        for i in range(len(labels)):
            if labels[i] == "background":
                labels[i] = 0
            elif labels[i] == "objective":
                labels[i] = 1
            elif labels[i] == "method":
                labels[i] = 2
            elif labels[i] == "result":
                labels[i] = 3
            elif labels[i] == "other":
                labels[i] = 4
            else:
                print(sentences[i])

        return token_ids, labels, seq_len, mask

    # load dataset
    train = load_dataset(config.train_path)
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.test_path)
    return train, dev, test

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches  # 数据集
        self.n_batches = len(batches) // self.batch_size
        self.residue = False
        if len(batches) % batch_size != 0:
            self.residue = True
        self.index = 1
        self.device = device

    def _to_tensor(self, datas):
        # 转换为tensor 并 to(device)
        x = torch.tensor([_[0] for _ in datas]).to(self.device)  # 输入序列
        y = torch.tensor([_[1] for _ in datas]).to(self.device)  # 标签
        seq_len = torch.tensor([_[2] for _ in datas]).to(self.device)
        mask = torch.Tensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[(self.index-1) * self.batch_size: len(self.batches)]
            max_len = 0
            for i in range(len(batches)):
                if len(batches[i][1]) > max_len:
                    max_len = len(batches[i][1])
            for i in range(len(batches)):
                for j in range(max_len - len(batches[i][1])):
                    batches[i][1].append(-1)
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:  # 构建每一个batch
            max_len = 0
            batches = self.batches[(self.index-1) * self.batch_size: self.index * self.batch_size]
            for i in range(len(batches)):
                if len(batches[i][1]) > max_len:
                    max_len = len(batches[i][1])
            for i in range(len(batches)):
                if len(batches[i][1]) < max_len:
                    for j in range(max_len - len(batches[i][1])):
                        batches[i][1].append(-1)
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, *value, end="\n"):

        current = time.strftime("[%Y-%m-%d %H:%M:%S]")
        s = current
        for v in value:
            s += " " + str(v)
        s += end
        print(s, end="")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(s)
