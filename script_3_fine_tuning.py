from code.DatasetLoader import DatasetLoader
from code.MethodBertComp import GraphBertConfig
from code.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification
from code.ResultSaving import ResultSaving
from code.Settings import Settings
from code.MethodGraphBertGraphClustering import MethodGraphBertGraphClustering
import numpy as np
import torch
import json


#---- 'SpaceGraph' , 'Roomgraph', 'HouseGAN' ----

dataset_name = 'SpaceGraph'

np.random.seed(1)
torch.manual_seed(1)

if dataset_name == 'SpaceGraph':
    nclass = 14
    nfeature = 14
    ngraph = 2177
elif dataset_name == 'Roomgraph':
    nclass = 9
    nfeature = 8
    ngraph = 2076
elif dataset_name == 'HouseGAN':
    nclass = 8
    nfeature = 5
    ngraph = 143184


#---- Fine-Tuning Task 1: Graph Bert Node Classification (SpaceGraph, Roomgraph, and HouseGAN) ----
if 1:
    #---- hyper-parameters ----
    if dataset_name == 'Roomgraph':
        lr = 0.01
        k = 15
        max_epoch = 300 # 500 ---- do an early stop when necessary ----
    elif dataset_name == 'SpaceGraph':
        lr = 0.01
        k = 15
        max_epoch = 300 # 150 ---- do an early stop when necessary ----
    elif dataset_name == 'HouseGAN':
        k = 15
        lr = 0.01
        max_epoch = 300 #2000 # it takes a long epochs to get good results, sometimes can be more than 2000

    x_size = nfeature
    hidden_size = intermediate_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2
    y_size = nclass
    graph_size = ngraph
    residual_type = 'graph_raw'
    # --------------------------

    print('************ Start ************')
    print('GrapBert, dataset: ' + dataset_name + ', residual: ' + residual_type + ', k: ' + str(k) + ', hidden dimension: ' + str(hidden_size) +', hidden layer: ' + str(num_hidden_layers) + ', attention head: ' + str(num_attention_heads))
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True

    bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=y_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)
    method_obj = MethodGraphBertNodeClassification(bert_config)
    #---- set to false to run faster ----
    method_obj.spy_tag = True
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/NE-GraphBert/'
    result_obj.result_destination_file_name = dataset_name + '_' + str(num_hidden_layers)

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------


    method_obj.save_pretrained('./result/PreTrained_NE-GraphBert/' + dataset_name + '/node_classification_complete_model/')
    print('************ Finish ************')
#------------------------------------


#---- Fine-Tuning Task 2: Graph Bert Graph Clustering (Cora, Citeseer, and Pubmed) ----
if 0:
    #---- hyper-parameters ----
    if dataset_name == 'pubmed':
        lr = 0.001
        k = 30
        max_epoch = 500 # 500 ---- do an early stop when necessary ----
    elif dataset_name == 'cora':
        lr = 0.01
        k = 7
        max_epoch = 150 #150 # ---- do an early stop when necessary ----
    elif dataset_name == 'citeseer':
        k = 5
        lr = 0.001
        max_epoch = 2000 #2000 # it takes a long epochs to converge, probably more than 2000

    x_size = nfeature
    hidden_size = intermediate_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2
    y_size = nclass
    graph_size = ngraph
    residual_type = 'graph_raw'
    # --------------------------

    print('************ Start ************')
    print('GrapBert, dataset: ' + dataset_name + ', residual: ' + residual_type + ', k: ' + str(k) + ', hidden dimension: ' + str(hidden_size) +', hidden layer: ' + str(num_hidden_layers) + ', attention head: ' + str(num_attention_heads))
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True

    bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=y_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)
    method_obj = MethodGraphBertGraphClustering(bert_config)
    #---- set to false to run faster ----
    method_obj.cluster_number = y_size
    method_obj.spy_tag = True
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/NE-GraphBert/clustering_' + dataset_name
    result_obj.result_destination_file_name = '_' + ''

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------

    print('************ Finish ************')
#------------------------------------
