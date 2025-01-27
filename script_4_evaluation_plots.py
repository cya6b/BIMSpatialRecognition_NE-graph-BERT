import matplotlib.pyplot as plt
import seaborn as sns
import torch
from code.ResultSaving import ResultSaving
from code.EvaluateClustering import EvaluateClustering

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, matthews_corrcoef, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc, f1_score, precision_score, recall_score
import torch.nn.functional as F
import itertools
from code.base_class.dataset import dataset
from code.DatasetLoader import DatasetLoader
import numpy as np
import csv
from sklearn.preprocessing import label_binarize

#---------- clustering results evaluation -----------------

dataset_name = 'SpaceGraph'

if 0:
    pre_train_task = 'node_reconstruction+structure_recovery'

    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/NE-GraphBert/'
    result_obj.result_destination_file_name = 'clustering_' + dataset_name + '_' + pre_train_task
    loaded_result = result_obj.load()

    eval_obj = EvaluateClustering()
    eval_obj.data = loaded_result
    eval_result = eval_obj.evaluate()

    print(eval_result)




#--------------- Graph Bert Pre-Training Records Convergence --------------

dataset_name = 'SpaceGraph'

if 0:
    if dataset_name == 'SpaceGraph':
        k = 15
    elif dataset_name == 'Roomgraph':
        k = 15
    elif dataset_name == 'HouseGAN':
        k = 15

    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/NE-GraphBert/'
    best_score = {}

    result_obj.result_destination_file_name = dataset_name + '_' + str(k) + '_node_reconstruction'
    record_dict = result_obj.load()
    epoch_list = sorted(record_dict.keys())
    print(record_dict)

    plt.figure(figsize=(8, 8))
    train_loss = [record_dict[epoch]['loss_train'] for epoch in epoch_list]
    plt.plot(epoch_list, train_loss, label='NE-Graph-Bert')

    plt.xlim(0, 300)
    plt.ylabel("training loss")
    plt.xlabel("epoch (iter. over data set)")
    plt.legend(loc="upper right")
    plt.show()



#--------------- Graph Bert Learning Convergence --------------

def moving_average(data_set, periods=7):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

dataset_name = 'SpaceGraph'

# if 1:
#     residual_type = 'graph_raw'
#     diffusion_type = 'sum'
#     # depth_list = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]#, 2, 3, 4, 5, 6, 9, 19, 29, 39, 49]
#     depth_list = [2, 9, 19, 29]
#     result_obj = ResultSaving('', '')
#     result_obj.result_destination_folder_path = './result/GraphBert/'
#     best_score = {}
#
#     depth_result_dict = {}
#     for depth in depth_list:
#         result_obj.result_destination_file_name = dataset_name + '_' + str(depth)
#         print(result_obj.result_destination_file_name)
#         depth_result_dict[depth] = result_obj.load()
#     print(depth_result_dict)
#
#     x = range(300)
#
#     plt.figure(figsize=(8, 8))
#     for depth in depth_list:
#         print(depth_result_dict[depth].keys())
#         train_acc = [depth_result_dict[depth][i]['acc_train'] for i in x]
#         plt.plot(x, train_acc, label='Bert-GE(' + str(depth) + '-layer)')
#
#     plt.xlim(0, 300)
#     plt.ylabel("training accuracy %")
#     plt.xlabel("epoch (iter. over training set)")
#     plt.legend(loc="lower right", fontsize='small')
#     plt.show()
#
#     plt.figure(figsize=(8, 8))
#     for depth in depth_list:
#         test_acc = [depth_result_dict[depth][i]['acc_test'] for i in x]
#         plt.plot(x, test_acc, label='BERT-GE(' + str(depth) + '-layer)')
#         best_score[depth] = max(test_acc)
#
#     plt.xlim(0, 300)
#     plt.ylabel("testing accuracy %")
#     plt.xlabel("epoch (iter. over training set)")
#     plt.legend(loc="lower right", fontsize='small')
#     plt.show()
#
#     print(best_score)

if 0:
    residual_type = 'graph_raw'
    diffusion_type = 'sum'
    depth_list = [2, 9, 19, 29]
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/NE-GraphBert/'
    best_score = {}

    depth_result_dict = {}
    for depth in depth_list:
        result_obj.result_destination_file_name = dataset_name + '_' + str(depth)
        print(result_obj.result_destination_file_name)
        depth_result_dict[depth] = result_obj.load()
    print(depth_result_dict)

    x = range(300)

    plt.figure(figsize=(8, 8),dpi=500)
    for depth in depth_list:
        print(depth_result_dict[depth].keys())
        train_acc = [depth_result_dict[depth][i]['acc_train'] for i in x]


        def exponential_moving_average(data_set, alpha=0.2):
            ema = [data_set[0]]  # 初始值
            for value in data_set[1:]:
                ema.append(alpha * value + (1 - alpha) * ema[-1])
            return ema


        train_acc_smooth = exponential_moving_average(train_acc)
        # train_acc_smooth = moving_average(train_acc)
        plt.plot(x[:len(train_acc_smooth)], train_acc_smooth, label='NE-Graph-BERT(' + str(depth) + '-layer)')

    plt.xlim(0, 300)
    plt.ylabel("training accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right", fontsize='small')
    plt.show()

    plt.figure(figsize=(8, 8),dpi=500)
    for depth in depth_list:
        test_acc = [depth_result_dict[depth][i]['acc_test'] for i in x]
        test_acc_smooth = exponential_moving_average(test_acc)
        plt.plot(x[:len(test_acc_smooth)], test_acc_smooth, label='NE-Graph-BERT(' + str(depth) + '-layer)')
        best_score[depth] = max(test_acc_smooth)

    plt.xlim(0, 300)
    plt.ylabel("testing accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right", fontsize='small')
    plt.show()

    print(best_score)

    residual_type = 'graph_raw'
    diffusion_type = 'sum'
    depth_list = [2, 9, 19, 29]
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/NE-GraphBert/'
    best_score = {}

    depth_result_dict = {}
    for depth in depth_list:
        result_obj.result_destination_file_name = dataset_name + '_' + str(depth)
        print(result_obj.result_destination_file_name)
        depth_result_dict[depth] = result_obj.load()
    print(depth_result_dict)

    x = range(300)

    plt.figure(figsize=(8, 8), dpi=500)
    for depth in depth_list:
        print(depth_result_dict[depth].keys())
        train_loss = [depth_result_dict[depth][i]['loss_train'] for i in x]
        train_loss_smooth = moving_average(train_loss)
        plt.plot(x[:len(train_loss_smooth)], train_loss_smooth, label='NE-Graph-BERT(' + str(depth) + '-layer)')

    plt.xlim(0, 300)
    plt.ylabel("training loss")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="upper right", fontsize='small')
    plt.show()

    plt.figure(figsize=(8, 8), dpi=500)
    for depth in depth_list:
        test_loss = [depth_result_dict[depth][i]['loss_test'] for i in x]
        test_loss_smooth = moving_average(test_loss)
        plt.plot(x[:len(test_loss_smooth)], test_loss_smooth, label='NE-Graph-BERT(' + str(depth) + '-layer)')
        best_score[depth] = min(test_loss_smooth)

    plt.xlim(0, 300)
    plt.ylabel("testing loss")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="upper right", fontsize='small')
    plt.show()

    print(best_score)

    # 选择特定的depth
    selected_depth = 2
    epoch_list = range(300)  # 添加这行代码来定义epoch_list

    # 创建一个新的CSV文件
    with open('epoch_loss_depth_2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入列名
        writer.writerow(["Epoch", "Loss"])
        # 写入每个epoch和对应的loss值
        if selected_depth in depth_result_dict:
            for i in range(len(epoch_list)):
                writer.writerow([epoch_list[i], depth_result_dict[selected_depth][i]['loss_train']])



#--------------- Graph Bert Learning Convergence 2 --------------

dataset_name = 'SpaceGraph'

accuracy = []
tp = []
tn = []
fp = []
fn = []
prf = []
sensitivity = []
specificity = []
mcc = []
roc = []
pr = []

if 1:
    residual_type = 'graph_raw'
    diffusion_type = 'sum'
    hidden_layers = 2
    # depth_list = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]#, 2, 3, 4, 5, 6, 9, 19, 29, 39, 49]
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/NE-GraphBert/'
    best_score = {}

    depth_result_dict = {}
    y_true = []
    y_pred = []
    depth = 2

    result_obj.result_destination_file_name = dataset_name + '_' + str(depth)
    # print(result_obj.result_destination_file_name)
    depth_result_dict[depth] = result_obj.load()
    # print(depth_result_dict)

    x = range(300)

    test_acc = [depth_result_dict[depth][i]['acc_test'] for i in x]
    # print(test_acc, '\t' , len(test_acc))

    bestEpoch = 0
    for i in range(1, len(test_acc)):
        if test_acc[bestEpoch] < test_acc[i]:
            bestEpoch = i
    print('best epoch : ', bestEpoch)
    if 'test_acc_data' in depth_result_dict[depth][bestEpoch]:
        test_acc_data = depth_result_dict[depth][bestEpoch]['test_acc_data']
        y_true = test_acc_data['true_y'].tolist()
        y_pred = test_acc_data['pred_y'].tolist()
    else:
        print("Key 'test_acc_data' not found in the dictionary.")
        # 或者你可以给 test_acc_data 一个默认值
        # test_acc_data = default_value
    test_op = depth_result_dict[depth][bestEpoch]['test_op']  # .tolist()
    # print(test_op[0], '\t' , len(test_op))
    probs = F.softmax(test_op, dim=1)
    probs = probs.max(1)[1]
    probs = probs.tolist()
    best_score[depth] = max(test_acc)

    ############################ Evaluation Metrices #######################
    '''# accuracy: (tp + tn) / (p + n)
    _accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy: %f' % _accuracy)

    # specificity

    # mcc score

    # precision-recall curve
    # prediction values 

    # precision tp / (tp + fp)
    precision = precision_score(y_true, y_pred)
    print('Precision: %f' % precision)

    # recall: tp / (tp + fn)
    recall = recall_score(y_true, y_pred)
    print('Recall: %f' % recall)

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, y_pred)
    print('F1 score: %f' % f1)

    # ROC AUC
    ## AUC -> pass probability value instead of y_pred
    auc = roc_auc_score(y_true, top_p)
    print('ROC AUC: %f' % auc)

    #print('y_test score = ',y_test)'''
    print('\n\nEvaluation Metrices :\n')

    y_test = y_true
    y_pred_nn = probs
    y_pred_nn1 = y_pred
    acc = accuracy_score(y_test, y_pred_nn1)
    accuracy.append(acc)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_nn1)
    data_obj = DatasetLoader()
    data_obj.dataset_name = 'SpaceGraph'  # 或者你正在使用的数据集名称
    data_obj.dataset_source_folder_path = './data/SpaceGraph/'  # 设置数据集文件的路径
    data_obj.load()
    class_names = data_obj.class_names

    cm = confusion_matrix(y_test, y_pred_nn1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    # 添加标题和轴标签
    plt.figure(figsize=(16, 13))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # 显示图像
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion matrix")
    plt.show()

    label_counts = {label: 0 for label in range(14)}
    total_instances = len(y_true)
    for label in y_true:
        label_counts[label] += 1
    label_support = {label: count / total_instances for label, count in label_counts.items()}

    for i in range(14):  # 你有14个类别
        print(f"{class_names[i]}: Precision: {precision[i]}, Recall: {recall[i]}, F1: {f1[i]}, Support: {support[i]}")
    # 计算TP, TN, FP, FN
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (FP + FN + TP)

    # 计算总体的TP, TN, FP, FN
    total_TP = np.sum(TP)
    total_TN = np.sum(TN)
    total_FP = np.sum(FP)
    total_FN = np.sum(FN)

    # 计算总体的sensitivity和specificity
    total_sensitivity = total_TP / (total_TP + total_FN)
    total_specificity = total_TN / (total_TN + total_FP)

    # 计算sensitivity和specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # 计算总体的Precision, Recall, F1
    total_precision = total_TP / (total_TP + total_FP)
    total_recall = total_TP / (total_TP + total_FN)
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    macro_f1 = np.mean(f1)
    # tn1, fp1, fn1, tp1 = confusion_matrix(y_test, y_pred_nn1).ravel()
    # sens = tp1 / (tp1 + fn1)
    # spec = tn1 / (tn1 + fp1)
    # prf1 = precision_recall_fscore_support(y_test, y_pred_nn1)
    # tp.append(tp1)
    # tn.append(tn1)
    # fp.append(fp1)
    # fn.append(fn1)
    # prf.append(prf1)
    # sensitivity.append(sens)
    # specificity.append(spec)
    # mathew = matthews_corrcoef(y_test, y_pred_nn1)
    # mcc.append(mathew)
    # auroc = roc_auc_score(y_test, y_pred_nn, multi_class='ovo')
    # roc.append(auroc)
    # aupr = average_precision_score(y_test, y_pred_nn)
    # pr.append(aupr)

    # ---- ROC ----
    # # 将标签二值化
    # y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    # y_pred_bin = label_binarize(y_pred_nn1, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    #
    # n_classes = y_test_bin.shape[1]
    #
    # # 计算每个类别的ROC曲线和AUC
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # 绘制所有类别的ROC曲线
    # plt.figure()
    # colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'blue']
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=2,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(class_names[i], roc_auc[i]))
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic to Multi-Class')
    # plt.legend(loc="lower right")
    # plt.show()
    # 将标签二值化
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    n_classes = y_test_bin.shape[1]

    # # 计算每个类别的ROC曲线和AUC
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], test_op[:, i].tolist())
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # 绘制所有类别的ROC曲线
    # plt.figure()
    # colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan',
    #           'magenta', 'yellow', 'black', 'blue']
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=2,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(class_names[i], roc_auc[i]))
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic to Multi-Class')
    # plt.legend(loc="lower right")
    # plt.show()

    # ----------------------------------------

    # ---- accuracy ----
    # 加载保存的结果
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/NE-GraphBert/'
    result_obj.result_destination_file_name = dataset_name + '_' + str(depth)
    results = result_obj.load()

    # 获取每个epoch的accuracy值
    epochs = sorted(results.keys())
    # 假设我们有7个类别
    # class_accuracies = [[] for _ in range(7)]
    # for epoch in epochs:
    #     for class_idx in range(7):
    #         class_accuracies[class_idx].append(results[epoch]['test_acc_data'][class_idx])
    accuracies = [results[epoch]['acc_test'] for epoch in epochs]

    # 创建一个新的figure
    plt.figure(figsize=(10, 8))

    # 绘制accuracy曲线，设置线条的颜色、宽度和样式
    plt.plot(epochs, accuracies, color='darkblue', linewidth=2, linestyle='-')

    # 绘制每个类别的accuracy曲线，设置线条的颜色、宽度和样式
    # colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown']
    # for class_idx in range(7):
    #     plt.plot(epochs, class_accuracies[class_idx], color=colors[class_idx], linewidth=2, linestyle='-')

    # 添加网格线
    plt.grid(True, which='both', color='gray', linewidth=0.5, linestyle='--')

    # 添加标题和轴标签
    plt.title('Accuracy over Epochs', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)

    # 设置轴的字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 添加图例
    # class_names = data_obj.class_names
    # plt.legend(class_names, loc="lower right", fontsize=12)
    plt.legend(['Accuracy'], loc="lower right", fontsize=12)

    # 显示图形
    plt.show()

    # ----------------------------------------

    print(cm)
    print('accuracy', accuracy)
    print('sensitivity', sensitivity)
    print('specificity', specificity)
    print('Total sensitivity', total_sensitivity)
    print('Total specificity', total_specificity)
    print('Total Precision', total_precision)
    print('Total Recall', total_recall)
    print('Total F1', total_f1)
    print('Macro F1', macro_f1)
    # print(TP, TN, FP, FN)
    # print(total_TP,total_TN,total_FP,total_FN)
    # 计算每个类别的精确度、召回率、F1分数和支持度
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    # 打印macro指标
    print('Macro Accuracy: %f' % acc)
    print('Macro Precision: %f' % precision)
    print('Macro Recall: %f' % recall)
    print('Macro F1 score: %f' % f1)
    # 计算总体的support
    total_support = np.sum(support)

    # 打印总体的support
    print('Total support:', total_support)


