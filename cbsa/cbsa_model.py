import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# class Config(object):
#     """配置参数"""
#
#     def __init__(self, dataset, model_name, pretrained_dataset):
#         self.model_name = model_name
#         self.translate = translate
#         self.dataset = dataset
#         self.pretrained_dataset = pretrained_dataset
#         # 训练集、验证集、测试集路径
#         self.train_path = './../data/' + self.dataset + '/train.jsonl'
#         self.dev_path = './../data/' + self.dataset + '/dev.jsonl'
#         self.test_path = './../data/' + self.dataset + '/test.jsonl'
#         # # 类别名单
#         # self.class_list = [x.strip() for x in open(
#         #     dataset + '/data/class.txt').readlines()]
#         self.class_list = ['background', 'objective', 'method', 'result', 'other']
#         #self.class_weights = [0.6, 1.2, 0.2, 0.4, 1.2]
#         #  + "-" +  self.dataset
#         # 存储模型的训练结果
#         self.save_path1 = './'+ self.model_name + "-" + self.pretrained_dataset + "/"
#         self.save_path2 = './save_model/' + self.model_name + "-" + self.pretrained_dataset + '.pt'
#         # self.save_path1 = './'+ self.model_name + "-" + self.pretrained_dataset + "-" +  self.dataset +"/"
#         # self.save_path2 = './save_model/' + self.model_name + "-" + self.pretrained_dataset + "-" +  self.dataset + '.pt'
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
#         self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
#         self.num_classes = 5 # 类别数
#         self.num_epochs = 5  # epoch数
#         self.batch_size = 4  # mini-batch大小
#         self.pad_size = 512  # 每句话处理成的长度(短填长切)
#         self.learning_rate = 5e-6  # 学习率 mBert
#         # self.learning_rate = 1e-5  # 学习率 roBerta
#
#         # 预训练模型相关文件(模型文件.bin、配置文件.json、词表文件vocab.txt)存储路径
#         # self.bert_path = 'models'
#         # 序列切分工具
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         # 隐藏单元数
#         # self.hidden_size = 119547  # mbert
#         # self.hidden_size = 250002  # xml-roberta
#         # self.hidden_size = 28996     #bert
#         self.hidden_size = 31116  # scibert-cased

class Config(object):
    """配置参数"""

    def __init__(self, opt):
        self.model_name = opt.model_name
        self.dataset = opt.dataset
        self.pretrained_dataset = opt.pretrained_dataset
        self.train = opt.train
        self.train_path = './../data/' + self.dataset + '/train.jsonl'
        self.dev_path = './../data/' + self.dataset + '/dev.jsonl'
        self.test_path = './../data/' + self.dataset + '/test.jsonl'
        self.class_list = ['background', 'objective', 'method', 'result', 'other']
        # the path for saving pretrained model
        if self.dataset == self.pretrained_dataset:
            self.save_path1 = './'+ self.model_name + "-" + self.pretrained_dataset + "/"
            self.save_path2 = './save_model/' + self.model_name + "-" + self.pretrained_dataset + '.pt'
            self.learning_rate = 5e-6  # 学习率 mBert
        else:
            self.save_path1 = './'+ self.model_name + "-" + self.pretrained_dataset + "-" +  self.dataset +"/"
            self.save_path2 = './save_model/' + self.model_name + "-" + self.pretrained_dataset + "-" +  self.dataset + '.pt'
            self.learning_rate = 1e-5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.require_improvement = 1000
        self.num_classes = 5
        self.num_epochs = 5
        self.batch_size = 4
        self.pad_size = 512
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model_name == "bert-base-multilingual-cased":
            self.hidden_size = 119547
        elif self.model_name == "xlm-roberta-base":
            self.hidden_size = 250002
        elif self.model_name == "allenai/scibert_scivocab_cased":
            self.hidden_size = 31116

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        if config.pretrained_dataset == config.dataset:
            self.bert = AutoModelForMaskedLM.from_pretrained(config.model_name)
        else:
            self.bert = AutoModelForMaskedLM.from_pretrained("./" + config.model_name + "-" + config.pretrained_dataset) # 本地加载模型
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = torch.nn.Dropout(p=0.1)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0][0]  # 输入的句子
        labels = x[1]
        masks = x[0][2]

        embedded_sentences = self.bert(context, attention_mask=masks)[0]
        if self.config.model_name == "xlm-roberta-base":
            sentences_mask = context == 2
        else:
            sentences_mask = context == 102
        embedded_sentences = embedded_sentences[sentences_mask]
        num_sentences = embedded_sentences.shape[0]
        batch_size = 1
        embedded_sentences = embedded_sentences.unsqueeze(dim=0)
        labels_mask = labels != -1
        labels = labels[labels_mask]
        num_labels = labels.shape[0]
        if num_labels != num_sentences:  # bert truncates long sentences, so some of the SEP tokens might be gone
            if num_labels < num_sentences:
                print(self.config.tokenizer.decode(context))
            assert num_labels > num_sentences  # but `num_labels` should be at least greater than `num_sentences`
            labels = labels[:num_sentences]  # Ignore some labels. This is ok for training but bad for testing.
            # We are ignoring this problem for now.
        labels = labels.unsqueeze(dim=0)
        labels_logits = self.fc(embedded_sentences)
        if labels is not None:
            flattened_logits = labels_logits.reshape((batch_size * num_sentences), 5).squeeze()
            flattened_gold = labels.contiguous().reshape(-1)
            label_loss = self.loss(flattened_logits, flattened_gold)
            label_loss = label_loss.mean()
        return flattened_logits, label_loss, labels
