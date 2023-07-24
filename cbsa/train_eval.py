import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from utils import build_iterator
from transformers import AdamW, get_linear_schedule_with_warmup

def train(logger, config, model, train_data, dev_data, test_data):

    logger.log()
    logger.log("dataset_name:", config.dataset)
    logger.log("pretrained dataset:", config.pretrained_dataset)
    logger.log("model_name:", config.model_name)
    logger.log("Device:", config.device)
    logger.log("Epochs:", config.num_epochs)
    logger.log("Batch Size:", config.batch_size)
    logger.log("Learning Rate:", config.learning_rate)
    logger.log("Max Sequence Length:", config.pad_size)
    logger.log()

    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer  = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    total_step = len(train_data) * config.num_epochs
    num_warmup_steps = round(total_step * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_step)

    model.to(config.device)
    total_batch = 0  # 记录进行到多少batch
    train_losses = []
    train_acces = []
    dev_losses = []
    dev_acces = []
    dev_best_loss = float('inf')
    if config.train:
        for epoch in range(config.num_epochs):
           train_iter = build_iterator(train_data, config)
           print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
           train_loss = 0
           train_acc = 0
           for i, trains in enumerate(train_iter):
               outputs, loss, labels = model(trains)
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               scheduler.step()
               # 训练损失
               train_loss += loss
               true = labels.data.view(-1).cpu()
               predict_probs = torch.softmax(outputs, dim=-1)
               predic = torch.max(predict_probs, 1)[1].cpu()
               acc = metrics.f1_score(true, predic, average="macro")
               train_acc += acc
               if total_batch % 100 == 0:
                   dev_iter = build_iterator(dev_data, config)
                   acc2, loss2 = evaluate(config, model, dev_iter)
                   if loss2 < dev_best_loss:
                       dev_best_loss = loss2
                       model.bert.save_pretrained(config.save_path1)
                       torch.save(model.state_dict(), config.save_path2)
                       improve = '*'
                       last_improve = total_batch
                   else:
                       improve = ''
                   time_dif = time.time() - start_time
                   msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5}'
                   logger.log(msg.format(total_batch, loss.item(), acc, loss2, acc2, time_dif))
                   model.train()
               total_batch += 1
           dev_iter = build_iterator(dev_data, config)
           dev_acc, dev_loss = evaluate(config, model, dev_iter)
           train_losses.append(train_loss / len(train_iter))
           train_acces.append(train_acc / len(train_iter))
           dev_losses.append(dev_loss)
           dev_acces.append(dev_acc)
           msg2 = 'epoch: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}'
           logger.log(msg2.format(epoch, train_loss/len(train_iter), train_acc/len(train_iter), dev_loss, dev_acc))
           model.train()
    test_iter = build_iterator(test_data, config)
    test(config, model, test_iter)


def evaluate(config, model, data_iter, test=False):
    model.eval()  # 测试模式
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, texts in enumerate(data_iter):
            outputs, loss, labels = model(texts)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict_probs = torch.softmax(outputs, dim=-1)
            predic = torch.max(predict_probs, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.f1_score(labels_all, predict_all, average="micro")
    if test:  # 如果是测试集的话 计算一下分类报告
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        metrics.ConfusionMatrixDisplay.from_predictions(labels_all, predict_all, display_labels=['background','objective', 'method', 'result', 'other'], cmap=plt.cm.Blues, normalize="true")
        plt.subplots_adjust(left=0.18, bottom=0.15)
        plt.show(cmap="Blues")
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

def test(logger, config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path2), False)
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    logger.log(msg.format(test_loss, test_acc))
    logger.log("Precision, Recall and F1-Score...")
    logger.log(test_report)
    logger.log("Confusion Matrix...")
    logger.log(test_confusion)
    time_dif = time.time() - start_time
    logger.log("Time usage:", time_dif)
