'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.optim as optim
from keras.layers import Input, Dense
from keras.models import Sequential, model_from_config,Model
from keras.layers.core import  Dropout, Activation, Flatten#, Merge
import numpy as np
from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')
from time import time

from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import scipy.io


def DNN_auto1(x_train):
    encoding_dim = 64  # 128 original
    input_img = Input(shape=(585,))

    encoded = Dense(350, activation='relu')(input_img)  # 450 - output (input layer)
    # encoded = Dense(250, activation='relu')(encoded)     # 200 - output (hidden layer1)
    encoded = Dense(150, activation='relu')(encoded)  # 100 - output (hidden layer2)
    encoder_output = Dense(encoding_dim)(encoded)  # 128 - output (encoding layer)
    print()
    # decoder layers
    decoded = Dense(150, activation='relu')(encoder_output)
    # decoded = Dense(250, activation='relu')(decoded)
    decoded = Dense(350, activation='relu')(decoded)
    decoded = Dense(585, activation='tanh')(decoded)

    autoencoder = Model(input=input_img, output=decoded)

    encoder = Model(input=input_img, output=encoder_output)

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(x_train, x_train, epochs=1000, batch_size=100,
                    shuffle=True)  # second x_train is given instead of train labels in DNN, ie here, i/p=o/p

    # batch_size=100 original
    encoded_imgs = encoder.predict(x_train)

    return encoder_output, encoded_imgs
def DNN_auto2(x_train):
    encoding_dim = 64  # 128 original
    input_img = Input(shape=(88,))
    encoder_output = Dense(encoding_dim)(input_img)  # 128 - output (encoding layer)
    print()
    # decoder layers
    decoded = Dense(88, activation='relu')(encoder_output)
    autoencoder = Model(input=input_img, output=decoded)

    encoder = Model(input=input_img, output=encoder_output)

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(x_train, x_train, epochs=1000, batch_size=15,
                    shuffle=True)  # second x_train is given instead of train labels in DNN, ie here, i/p=o/p

    # batch_size=100 original
    encoded_imgs = encoder.predict(x_train)

    return encoder_output, encoded_imgs

if __name__ == '__main__':

    args.device = torch.device('cuda:' + str(args.gpu_id))

    plain_adj, norm_adj, mean_adj = data_generator[0].get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)


    # 自动编码器
    # circRNA_fea = np.loadtxt('../Data/circ_dis/integrated CircRNA Similarity.csv', dtype=float, delimiter=",")
    # disease_fea = np.loadtxt('../Data/circ_dis/integrated Disease Similarity.csv', dtype=float, delimiter=",")
    # _, circRNA_Embed = DNN_auto1(circRNA_fea)
    # _, disease_Embed = DNN_auto2(disease_fea)
    # np.savetxt('../Data/circ_dis/ciRNAEmbedding.csv', circRNA_Embed, delimiter=',')
    # np.savetxt('../Data/circ_dis/diseaseEmbedding.csv', disease_Embed, delimiter=',')
    # print()

    interaction = np.zeros((673, 673))
    inter = np.loadtxt("../Data/circ_dis/Association matrix.csv",dtype=float,delimiter=",")
    interaction[88:, :88] = inter
    interaction[:88, 88:] = inter.T

    model = NGCF(data_generator[0].n_users,
                 data_generator[0].n_items,
                 norm_adj,
                 interaction,
                 args).to(args.device)

    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    # for epoch in range(args.epoch):
    #     t1 = time()
    #     loss, mf_loss, emb_loss = 0., 0., 0.
    #     n_batch = data_generator[0].n_train // args.batch_size + 1
    #
    #     for idx in range(n_batch):
    #         users, pos_items, neg_items = data_generator[0].sample()
    #         pos_output, neg_output = model(users, pos_items, neg_items, drop_flag=args.node_dropout_flag)
    #         batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(pos_output, neg_output)
    #         optimizer.zero_grad()
    #         batch_loss.backward()
    #         optimizer.step()
    #
    #         loss += batch_loss
    #         mf_loss += batch_mf_loss
    #         emb_loss += batch_emb_loss
    #
    #     if (epoch + 1) % 10 != 0:
    #         if args.verbose > 0 and epoch % args.verbose == 0:
    #             perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
    #                 epoch, time() - t1, loss, mf_loss, emb_loss)
    #             print(perf_str)
    #         continue
    #
    #     t2 = time()
    #     users_to_test = list(data_generator[0].test_set.keys())
    #     ret = test(0, model, users_to_test, drop_flag=False)
    #
    #     pyplot.show()
    #     t3 = time()
    #
    #     loss_loger.append(loss)
    #     rec_loger.append(ret['recall'])
    #     pre_loger.append(ret['precision'])
    #     ndcg_loger.append(ret['ndcg'])
    #     hit_loger.append(ret['hit_ratio'])
    #
    #     if args.verbose > 0:
    #         perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
    #                    'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f], auc=[%.5f]' % \
    #                    (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
    #                     ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
    #                     ret['ndcg'][0], ret['ndcg'][-1], ret['auc'])
    #         print(perf_str)
    #
    #     cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
    #                                                                 stopping_step, expected_order='acc', flag_step=5)
    #
    #     # *********************************************************
    #     # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
    #     # if should_stop == True:
    #     #     break
    #
    #     # *********************************************************
    #     # save the user & item embeddings for pretraining.
    #     if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
    #         torch.save(model.state_dict(), args.weights_path + str(epoch) + 'C.pkl')
    #         print('save the weights in path: ', args.weights_path + str(epoch) + 'C.pkl')
    #
    # recs = np.array(rec_loger)
    # pres = np.array(pre_loger)
    # ndcgs = np.array(ndcg_loger)
    # hit = np.array(hit_loger)
    #
    # best_rec_0 = max(recs[:, 0])
    # idx = list(recs[:, 0]).index(best_rec_0)
    #
    # final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
    #              (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
    #               '\t'.join(['%.5f' % r for r in pres[idx]]),
    #               '\t'.join(['%.5f' % r for r in hit[idx]]),
    #               '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    # print(final_perf)



    """
    ***************************************************
    Test
    """

    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    loss, mf_loss, emb_loss = 0., 0., 0.
    for i in range(5):
        generator = data_generator[i+1]
        n_batch = generator.n_train // args.batch_size + 1
        users, pos_items, neg_items = generator.sample()
        model.load_state_dict(torch.load("./model/49B.pkl"))

        users_to_test = list(generator.test_set.keys())
        ret = test(i+1, model, users_to_test, drop_flag=False, is_test=True)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f], auc=[%.5f] , aupr=[%.5f], acc=[%.5f]' % \
                       (i, 1, 1, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1], ret['auc'], ret['aupr'], ret['acc'])
            print(perf_str)


    pyplot.show()
    print()
    # """
    # ************************************************
    # Prediction
    # """
    # cur_best_pre_0, stopping_step = 0, 0
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #
    # loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    # loss, mf_loss, emb_loss = 0., 0., 0.
    # n_batch = data_generator[0].n_train // args.batch_size + 1
    # users, pos_items, neg_items = data_generator[0].sample()
    # model.load_state_dict(torch.load("./model/79A.pkl"))
    #
    #
    # users_to_pred = list(data_generator[0].pred_items.keys())
    # ret = prediction(model, users_to_pred, drop_flag=False)
    #
    #
    # loss_loger.append(loss)
    # rec_loger.append(ret['recall'])
    # pre_loger.append(ret['precision'])
    # ndcg_loger.append(ret['ndcg'])
    # hit_loger.append(ret['hit_ratio'])
    #
    # if args.verbose > 0:
    #     perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
    #                'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f], auc=[%.5f]' % \
    #                (1, 1, 1, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
    #                 ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
    #                 ret['ndcg'][0], ret['ndcg'][-1], ret['auc'])
    #     print(perf_str)
