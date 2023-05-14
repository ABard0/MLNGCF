'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse


# new
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)

class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, interaction, args):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size
        self.conv1 = nn.Conv1d(64, 1, 1)
        self.conv2 = nn.Conv1d(64, 1, 1)
        self.conv3 = nn.Conv1d(673, 64, 1)
        self.LeakyRelu = nn.LeakyReLU(inplace=True)
        self.softmax = nn.Softmax()

        # collaborative filtering part
        self.dense1 = nn.Linear(512, 256)
        self.conv1d1 = nn.Conv1d(512, 256, 1, stride=1)
        self.conv1d2 = nn.Conv1d(256, 256, 1, stride=1)
        self.conv1d3 = nn.Conv1d(256, 256, 1, stride=1)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(256, 256)

        self.denseFin1 = nn.Linear(256, 1)
        self.Conv1d_1 = nn.Conv1d(256, 1, 1, stride=1)
        # self.denseFin2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

        # input = torch.randn(1, 10, 256)
        # input = input.permute(0, 2, 1)
        # m = nn.Conv1d(256, 1, 1, stride=1)
        # output = m(input)

        self.norm_adj = norm_adj
        self.interaction = interaction

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]
        self.alph = 0.5
        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_
        '../Data/circ_dis'
        circRNA_fea = np.loadtxt('../Data/circ_dis/ciRNAEmbedding.csv', dtype=float, delimiter=",")
        disease_fea = np.loadtxt('../Data/circ_dis/diseaseEmbedding.csv', dtype=float, delimiter=",")

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.tensor(circRNA_fea).to(torch.float32))),
            'item_emb': nn.Parameter(initializer(torch.tensor(disease_fea).to(torch.float32)))
        })

        # embedding_dict = nn.ParameterDict({
        #     'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
        #                                          self.emb_size))),
        #     'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
        #                                          self.emb_size)))
        # })


        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))


    def create_bpr_loss(self, pos_output, neg_output):
        # pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        # neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        pos_scores = torch.sum(pos_output)
        neg_scores = torch.sum(neg_output)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(pos_output) ** 2
                       + torch.norm(neg_output) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, pos_output):
        matrix = torch.zeros((int(len(pos_output)/585), 585))
        k = 0
        for i in range(int(len(pos_output)/585)):
            for j in range(585):
                matrix[i][j] = pos_output[k]
                k = k+1
        return matrix


    def forward(self, users, pos_items, neg_items, drop_flag=True, is_train=True):

        def feature_normalize(data):
            mu = torch.mean(data, dim=0)
            std = torch.std(data, dim=0)
            return (data - mu) / std

        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]
        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k]

            # multi-head Attention
            seq_fts = bi_embeddings.unsqueeze(0)
            seq_fts = seq_fts.view(1, 64, 673)
            f_1 = self.conv1(seq_fts)
            f_2 = self.conv1(seq_fts)
            f_1 = torch.transpose(f_1, 1, 2)

            logits = f_1 + torch.transpose(f_2, 1, 2)

            adj = torch.from_numpy(self.interaction).unsqueeze(0)
            bias_mat = torch.from_numpy(adj_to_bias(adj, [self.n_user + self.n_user], 1)).cuda()
            coefs = self.softmax(self.LeakyRelu(logits[0]) + bias_mat[0].type(torch.float32))
            # coefs = coefs.unsqueeze(0)
            # coefs = self.conv3(coefs)[0]
            bi_embeddings = torch.matmul(coefs, torch.transpose(seq_fts[0], 0, 1))

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # # multi-head Attention
            # seq_fts = ego_embeddings.unsqueeze(0)
            # seq_fts = seq_fts.view(1, 64, 673)
            # f_1 = self.conv1(seq_fts)
            # f_2 = self.conv1(seq_fts)
            # f_1 = torch.transpose(f_1, 1, 2)
            #
            # logits = f_1 + torch.transpose(f_2, 1, 2)
            #
            # adj = torch.from_numpy(self.interaction).unsqueeze(0)
            # bias_mat = torch.from_numpy(adj_to_bias(adj, [self.n_user+self.n_user], 1)).cuda()
            # coefs = self.softmax(self.LeakyRelu(logits[0]) + bias_mat[0].type(torch.float32))
            # # coefs = coefs.unsqueeze(0)
            # # coefs = self.conv3(coefs)[0]
            # ego_embeddings = torch.matmul(coefs,  torch.transpose(seq_fts[0], 0, 1))


            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        """
        *********************************************************
        look up.
        """
        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]



        # collaborative filtering
        # Positive
        neg_output = []
        pos_output = []
        if is_train:
            # MF part
            mf_output = torch.mul(u_g_embeddings, pos_i_g_embeddings)
            #mf_output = F.normalize(mf_output)
            # MLP apart
            mlp_output = torch.cat((u_g_embeddings, pos_i_g_embeddings), dim=1)
            layer1 = self.conv1d1(mlp_output.unsqueeze(0).permute(0, 2, 1))
            layer1 = self.relu1(layer1)
            layer2 = self.conv1d2(layer1)
            layer2 = self.relu1(layer2)
            mlp_output = self.conv1d3(layer2)
            mlp_output = self.relu1(mlp_output)
            #layer1 = self.dense1(mlp_output)
            #layer1 = self.relu1(layer1)
            #layer2 = self.dense2(layer1)
            #layer2 = self.relu2(layer2)
            #mlp_output = self.dense3(layer2)
            #mlp_output = feature_normalize(mlp_output)
            mlp_output = mlp_output.permute(0, 2, 1).squeeze(0)
            pos_output = 0.001*mlp_output+mf_output
            pos_output = torch.sum(pos_output, dim=1)
            #pos_output = self.Conv1d_1(pos_output.unsqueeze(0).permute(0, 2, 1))
            #pos_output = self.denseFin1(pos_output)
            #pos_output = pos_output.permute(0, 2, 1).squeeze(0)
            #pos_output = self.denseFin2(pos_output)
            #pos_output = self.sigmoid(pos_output)


            # Negative
            mf_output = torch.mul(u_g_embeddings, neg_i_g_embeddings)
            #mf_output = feature_normalize(mf_output)
            # MLP *part
            mlp_output = torch.cat((u_g_embeddings, neg_i_g_embeddings), dim=1)
            layer1 = self.conv1d1(mlp_output.unsqueeze(0).permute(0, 2, 1))
            layer1 = self.relu1(layer1)
            layer2 = self.conv1d2(layer1)
            layer2 = self.relu1(layer2)
            mlp_output = self.conv1d3(layer2)
            #layer1 = self.dense1(mlp_output)
            #layer1 = self.relu1(layer1)
            #layer2 = self.dense2(layer1)
            #layer2 = self.relu2(layer2)
            #mlp_output = self.dense3(layer2)
           # mlp_output = feature_normalize(mlp_output)
            mlp_output = mlp_output.permute(0, 2, 1).squeeze(0)

            neg_output = 0.001*mlp_output+mf_output
            neg_output = torch.sum(neg_output, dim=1)

            #neg_output = self.denseFin1(neg_output)
            #neg_output = neg_output.permute(0, 2, 1).squeeze(0)
            #neg_output = self.denseFin2(neg_output)
            #neg_output = self.sigmoid(neg_output)

        else:
            mf_output = torch.zeros((len(u_g_embeddings)*len(pos_i_g_embeddings),256)).cuda()
            mlp_output = torch.zeros((len(u_g_embeddings)*len(pos_i_g_embeddings),512)).cuda()
            k = 0
            for i in range(len(u_g_embeddings)):
                tmp_u = u_g_embeddings[i]
                for j in range(len(pos_i_g_embeddings)):
                    mf_output[k] = torch.mul(tmp_u, pos_i_g_embeddings[j])
                    mlp_output[k] = torch.cat((tmp_u, pos_i_g_embeddings[j]))
                    k = k+1
            mf_output = feature_normalize(mf_output)
            #mlp_output = feature_normalize(mlp_output)
            #layer1 = self.dense1(mlp_output)
            #layer1 = self.relu1(layer1)
            #layer2 = self.dense2(layer1)
           # layer2 = self.relu2(layer2)
            #mlp_output = self.dense3(layer2)
            layer1 = self.conv1d1(mlp_output.unsqueeze(0).permute(0, 2, 1))
            layer1 = self.relu1(layer1)
            layer2 = self.conv1d2(layer1)
            layer2 = self.relu1(layer2)
            mlp_output = self.conv1d3(layer2)
            mlp_output = mlp_output.permute(0, 2, 1).squeeze(0)
            #mlp_output = feature_normalize(mlp_output)
            pos_output = 0.001*mlp_output+mf_output
            pos_output = torch.sum(pos_output, dim=1)
            #pos_output = self.denseFin1(pos_output)
            #pos_output = self.denseFin2(pos_output)
            #pos_output = pos_output.permute(0, 2, 1).squeeze(0)


            #pos_output = self.sigmoid(pos_output)




            print()
        # # MF part
        # mf_user_latent = Flatten()(MF_Embedding_User(user_input))
        # mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
        # mf_vector = merge([mf_user_latent, mf_item_latent], mode='mul')  # element-wise multiply
        #
        # # MLP part
        # mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
        # mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
        # mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode='concat')
        # for idx in xrange(1, num_layer):
        #     layer = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu', name="layer%d" % idx)
        #     mlp_vector = layer(mlp_vector)
        #
        # # Concatenate MF and MLP parts
        # # mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
        # # mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
        # predict_vector = merge([mf_vector, mlp_vector], mode='concat')
        #
        # # Final prediction layer
        # prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name="prediction")(predict_vector)


        return pos_output, neg_output

