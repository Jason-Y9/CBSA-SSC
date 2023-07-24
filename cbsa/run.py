import time
import torch
import numpy as np
import argparse
from utils import build_dataset
from utils import Logger
from importlib import import_module
from train_eval import train

def parse_opt():
    # 声明argparse对象 可附加说明
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    # 模型是必须设置的参数(required=True) 类型是字符串
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased", help="choose a model: bert-base-multilingual-cased, xlm-roberta-base,"
                                                                                               "allenai/scibert_scivocab_cased")
    parser.add_argument("--dataset", type=str, default="CBSA", help="choose a dataset: CBSA, CBSA_eng, PubMed, CSABSTRACT, CSABSTRACT+CBSA_eng, PubMed+CBSA_eng,"
                                                                    "PubMed+CSABSTRACT+CBSA_eng, CSABSTRACT+CBSA, PubMed+CBSA, PubMed+CSABSTRACT+CBSA, PuMed+CSABSTRACT")
    parser.add_argument("--pretrained_dataset", type=str, default="CBSA", help="choose a dataset for pretraining : CBSA, CBSA_eng, PubMed, CSABSTRACT, CSABSTRACT+CBSA_eng, PubMed+CBSA_eng,"
                                                                    "PubMed+CSABSTRACT+CBSA_eng, CSABSTRACT+CBSA, PubMed+CBSA, PubMed+CSABSTRACT+CBSA, PuMed+CSABSTRACT")
    parser.add_argument("--train", type=bool, default=True, help="training or not")

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    x = import_module("cbsa_model")
    config = x.Config(opt)
    # dataset = 'CBSA_eng'
    # #  xlm-roberta-base, bert-base-multilingual-cased, allenai/scibert_scivocab_cased
    # model_name = "allenai/scibert_scivocab_cased"
    # pretrained_dataset = "CBSA_eng"
    # config = x.Config(dataset, model_name, pretrained_dataset)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    time_dif = time.time()-start_time
    print("Time usage:", time_dif)


    # train
    if config.pretrained_dataset == config.dataset:
        loging_path = "save_model/" + config.model_name + "-" + config.pretrained_dataset + ".txt"
    else:
        if config.train:
            loging_path = "save_model/" + config.model_name + "-" + config.pretrained_dataset + "-" + config.dataset + ".txt"
        else:
            loging_path = "save_model/" + config.model_name + "-" + config.pretrained_dataset + "-" + config.dataset + "-test.txt"
    logger = Logger(loging_path)
    model = x.Model(config).to(config.device)  # 构建模型对象
    train(logger, config, model, train_data, dev_data, test_data)  # 训练
