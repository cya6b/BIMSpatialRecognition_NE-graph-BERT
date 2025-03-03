import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from code.MethodGraphBert import MethodGraphBert

import time
import numpy as np

from code.EvaluateAcc import EvaluateAcc


BertLayerNorm = torch.nn.LayerNorm

class MethodGraphBertNodeClassification(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.01
    weight_decay = 5e-4
    # weight_decay = 0.01
    max_epoch = 500
    spy_tag = True

    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config, dropout_rate=0.3):
        super(MethodGraphBertNodeClassification, self).__init__(config)
        self.config = config
        self.bert = MethodGraphBert(config)
        self.res_h = torch.nn.Linear(config.x_size, config.hidden_size)
        self.res_y = torch.nn.Linear(config.x_size, config.y_size)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.y_size)
        self.init_weights()
        self.dropout = torch.nn.Dropout(dropout_rate)
    #     self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.to(self._device)
    # @property
    # def device(self):
    #     return self._device

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, idx=None):
        residual_h, residual_y = self.residual_term()
        if idx is not None:
            if residual_h is None:
                outputs = self.bert(raw_features[idx], wl_role_ids[idx], init_pos_ids[idx], hop_dis_ids[idx], residual_h=None)
            else:
                outputs = self.bert(raw_features[idx], wl_role_ids[idx], init_pos_ids[idx], hop_dis_ids[idx], residual_h=residual_h[idx])
                residual_y = residual_y[idx]
        else:
            if residual_h is None:
                outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, residual_h=None)
            else:
                outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, residual_h=residual_h)

        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        sequence_output = self.dropout(sequence_output)

        labels = self.cls_y(sequence_output)

        if residual_y is not None:
            labels += residual_y

        # return F.log_softmax(labels, dim=1)
        return [labels, F.log_softmax(labels, dim=1)]

    # def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, idx=None):
    #     residual_h, residual_y = self.residual_term()
    #     if idx is not None:
    #         raw_features = raw_features[idx].to(self.device)
    #         wl_role_ids = wl_role_ids[idx].to(self.device)
    #         init_pos_ids = init_pos_ids[idx].to(self.device)
    #         hop_dis_ids = hop_dis_ids[idx].to(self.device)
    #         if residual_h is not None:
    #             residual_h = residual_h[idx].to(self.device)
    #             residual_y = residual_y[idx].to(self.device)
    #     else:
    #         raw_features = raw_features.to(self.device)
    #         wl_role_ids = wl_role_ids.to(self.device)
    #         init_pos_ids = init_pos_ids.to(self.device)
    #         hop_dis_ids = hop_dis_ids.to(self.device)
    #         if residual_h is not None:
    #             residual_h = residual_h.to(self.device)
    #             residual_y = residual_y.to(self.device)

    #     if residual_h is None:
    #         outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, residual_h=None)
    #     else:
    #         outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, residual_h=residual_h)

    #     sequence_output = 0
    #     for i in range(self.config.k + 1):
    #         sequence_output += outputs[0][:, i, :]
    #     sequence_output /= float(self.config.k + 1)

    #     sequence_output = self.dropout(sequence_output)

    #     labels = self.cls_y(sequence_output)

    #     if residual_y is not None:
    #         labels += residual_y

    #     return [labels, F.log_softmax(labels, dim=1)]

    def residual_term(self):
        if self.config.residual_type == 'none':
            return None, None
        elif self.config.residual_type == 'raw':
            return self.res_h(self.data['X']), self.res_y(self.data['X'])
        elif self.config.residual_type == 'graph_raw':
            return torch.spmm(self.data['A'], self.res_h(self.data['X'])), torch.spmm(self.data['A'], self.res_y(self.data['X']))

    # def residual_term(self):
    #     if self.config.residual_type == 'none':
    #         return None, None
    #     elif self.config.residual_type == 'raw':
    #         return self.res_h(self.data['X'].to(self.device)), self.res_y(self.data['X'].to(self.device))
    #     elif self.config.residual_type == 'graph_raw':
    #         return torch.spmm(self.data['A'].to(self.device), self.res_h(self.data['X'].to(self.device))), torch.spmm(
    #             self.data['A'].to(self.device), self.res_y(self.data['X'].to(self.device)))

    def train_model(self, max_epoch):
        # 设置随机种子
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        accuracy = EvaluateAcc('', '')

        # # Separate the weights
        # weight_decay_parameters = list(map(id, self.cls_y.parameters()))
        # base_parameters = filter(lambda p: id(p) not in weight_decay_parameters, self.parameters())
        #
        # # Create separate optimizers
        # optimizer_base = optim.Adam(base_parameters, lr=self.lr)
        # optimizer_weight_decay = optim.Adam(self.cls_y.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        max_score = 0.0
        for epoch in range(max_epoch):
            t_epoch_begin = time.time()

            # -------------------------

            self.train()
            optimizer.zero_grad()
            # optimizer_base.zero_grad()
            # optimizer_weight_decay.zero_grad()

            output = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'], self.data['hop_embeddings'], self.data['idx_train'])
            output = output[1]

            loss_train = F.cross_entropy(output, self.data['y'][self.data['idx_train']])
            accuracy.data = {'true_y': self.data['y'][self.data['idx_train']], 'pred_y': output.max(1)[1]}
            acc_train = accuracy.evaluate()

            loss_train.backward()
            optimizer.step()
            # optimizer_base.step()
            # optimizer_weight_decay.step()

            self.eval()
            output = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'], self.data['hop_embeddings'], self.data['idx_val'])
            output = output[1]

            loss_val = F.cross_entropy(output, self.data['y'][self.data['idx_val']])
            accuracy.data = {'true_y': self.data['y'][self.data['idx_val']],
                             'pred_y': output.max(1)[1]}
            acc_val = accuracy.evaluate()

            #-------------------------
            #---- keep records for drawing convergence plots ----
            output = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'], self.data['hop_embeddings'], self.data['idx_test'])
            test_op = output[0]
            output = output[1]

            loss_test = F.cross_entropy(output, self.data['y'][self.data['idx_test']])
            accuracy.data = {'true_y': self.data['y'][self.data['idx_test']],
                             'pred_y': output.max(1)[1]}
            acc_test = accuracy.evaluate()

            self.learning_record_dict[epoch] = {'loss_train': loss_train.item(), 'acc_train': acc_train.item(),
                                                'loss_val': loss_val.item(), 'acc_val': acc_val.item(),
                                                'loss_test': loss_test.item(), 'acc_test': acc_test.item(),
                                                'test_acc_data': accuracy.data,
                                                'test_op': test_op,
                                                'time': time.time() - t_epoch_begin}

            # -------------------------
            if epoch % 10 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'loss_test: {:.4f}'.format(loss_test.item()),
                      'acc_test: {:.4f}'.format(acc_test.item()),
                      'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin) + ', best testing performance {: 4f}'.format(np.max([self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict])) + ', minimun loss {: 4f}'.format(np.min([self.learning_record_dict[epoch]['loss_test'] for epoch in self.learning_record_dict])))
        return time.time() - t_begin, np.max([self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict])

    # def train_model(self, max_epoch):
    #     torch.manual_seed(0)
    #     torch.cuda.manual_seed_all(0)
    #     np.random.seed(0)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    #     t_begin = time.time()
    #     optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     accuracy = EvaluateAcc('', '')

    #     max_score = 0.0
    #     for epoch in range(max_epoch):
    #         t_epoch_begin = time.time()

    #         self.train()
    #         optimizer.zero_grad()

    #         output = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'],
    #                               self.data['hop_embeddings'], self.data['idx_train'])
    #         output = output[1]

    #         loss_train = F.cross_entropy(output, self.data['y'][self.data['idx_train']].to(self.device))
    #         accuracy.data = {'true_y': self.data['y'][self.data['idx_train']].to(self.device).cpu(),
    #                          'pred_y': output.max(1)[1].cpu()}
    #         acc_train = accuracy.evaluate()

    #         loss_train.backward()
    #         optimizer.step()

    #         self.eval()
    #         output = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'],
    #                               self.data['hop_embeddings'], self.data['idx_val'])
    #         output = output[1]

    #         loss_val = F.cross_entropy(output, self.data['y'][self.data['idx_val']].to(self.device))
    #         accuracy.data = {'true_y': self.data['y'][self.data['idx_val']].to(self.device).cpu(),
    #                          'pred_y': output.max(1)[1].cpu()}
    #         acc_val = accuracy.evaluate()

    #         output = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'],
    #                               self.data['hop_embeddings'], self.data['idx_test'])
    #         test_op = output[0]
    #         output = output[1]

    #         loss_test = F.cross_entropy(output, self.data['y'][self.data['idx_test']].to(self.device))
    #         accuracy.data = {'true_y': self.data['y'][self.data['idx_test']].to(self.device).cpu(),
    #                          'pred_y': output.max(1)[1].cpu()}
    #         acc_test = accuracy.evaluate()

    #         self.learning_record_dict[epoch] = {'loss_train': loss_train.item(), 'acc_train': acc_train.item(),
    #                                             'loss_val': loss_val.item(), 'acc_val': acc_val.item(),
    #                                             'loss_test': loss_test.item(), 'acc_test': acc_test.item(),
    #                                             'test_acc_data': accuracy.data,
    #                                             'test_op': test_op,
    #                                             'time': time.time() - t_epoch_begin}

    #         if epoch % 10 == 0:
    #             print('Epoch: {:04d}'.format(epoch + 1),
    #                   'loss_train: {:.4f}'.format(loss_train.item()),
    #                   'acc_train: {:.4f}'.format(acc_train.item()),
    #                   'loss_val: {:.4f}'.format(loss_val.item()),
    #                   'acc_val: {:.4f}'.format(acc_val.item()),
    #                   'loss_test: {:.4f}'.format(loss_test.item()),
    #                   'acc_test: {:.4f}'.format(acc_test.item()),
    #                   'time: {:.4f}s'.format(time.time() - t_epoch_begin))

    #     print("Optimization Finished!")
    #     print("Total time elapsed: {:.4f}s".format(time.time() - t_begin) + ', best testing performance {: 4f}'.format(
    #         np.max([self.learning_record_dict[epoch]['acc_test'] for epoch in
    #                 self.learning_record_dict])) + ', minimun loss {: 4f}'.format(
    #         np.min([self.learning_record_dict[epoch]['loss_test'] for epoch in self.learning_record_dict])))
    #     return time.time() - t_begin, np.max(
    #         [self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict])

    def run(self):

        self.train_model(self.max_epoch)

        return self.learning_record_dict
