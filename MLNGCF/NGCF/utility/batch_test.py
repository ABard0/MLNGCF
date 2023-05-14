'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import sklearn.metrics
import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
import multiprocessing
import heapq
from matplotlib import pyplot

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import roc_curve
import pandas as pd

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)

data_generator =[Data(path=args.data_path + args.dataset, batch_size=args.batch_size, k=0),
                 Data(path=args.data_path + args.dataset, batch_size=args.batch_size, k=1),
                 Data(path=args.data_path + args.dataset, batch_size=args.batch_size, k=2),
                 Data(path=args.data_path + args.dataset, batch_size=args.batch_size, k=3),
                 Data(path=args.data_path + args.dataset, batch_size=args.batch_size, k=4),
                 Data(path=args.data_path + args.dataset, batch_size=args.batch_size, k=5)]



USR_NUM, ITEM_NUM = data_generator[0].n_users, data_generator[0].n_items
N_TRAIN, N_TEST = data_generator[0].n_train, data_generator[0].n_test
BATCH_SIZE = args.batch_size

def ranklist_by_heapq3(u, pred_items, rating, Ks):
    item_score = {}
    for i in pred_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(30, item_score, key=item_score.get)
    diseaseList = []
    with open('../Data/circ_dis/diseaseList.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            diseaseList.append(line)
    circList = []
    with open('../Data/circ_dis/circRNAList.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            circList.append(line)
    pre_result = []
    for i in range(len(K_max_item_score)):
        circRNA = circList[K_max_item_score[i]]
        disease = diseaseList[u]
        if item_score[K_max_item_score[i]] > 1.8:
             pre_result.append([circRNA, disease, item_score[K_max_item_score[i]]])
    df = pd.DataFrame(pre_result)
    df.to_csv("./preResult/predictResult.csv", mode='a')
    return K_max_item_score


def ranklist_by_heapq2(user_pos_test, test_items, rating, Ks, u_batch_id, fold):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    auc, aupr, acc = get_auc2(fold, item_score, user_pos_test, u_batch_id)

    return r, auc, aupr, acc


def get_auc2(fold, item_score, user_pos_test, u_batch_id):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.AUC(ground_truth=r, prediction=posterior)
    fpr, tpr, auc_thresholds = roc_curve(r, posterior)
    newpost = []
    for i in posterior:
        if i < auc_thresholds[3]:
            newpost.append(0)
        else:
            newpost.append(1)
    acc = accuracy_score(newpost, r)
    precision, recall, thresholds = precision_recall_curve(r, posterior)
    aupr = sklearn.metrics.auc(recall, precision)
    # 绘制Recall曲线
    # pyplot.plot(recall, precision, label='ROC fold %d ' % fold)
    # pyplot.xlabel('recall')
    # pyplot.ylabel('precision')
    # pyplot.title('Precision - Recall curve')
    # 绘制ROC曲线
    pyplot.plot(fpr, tpr, label='ROC fold %d (AUC = %0.4f)' % (fold, auc))
    pyplot.xlabel('False positive rate, (1-Specificity)')
    pyplot.ylabel('True positive rate,(Sensitivity)')
    pyplot.title('Receiver Operating Characteristic curve: 5-Fold CV')
    pyplot.legend()
    #pyplot.show()
    return auc, aupr, acc


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.AUC(ground_truth=r, prediction=posterior)
    fpr, tpr, auc_thresholds = roc_curve(r, posterior)

    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, aupr, acc, Ks):
    precision, recall, ndcg, hit_ratio, f1_scoere = [], [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc, 'aupr': aupr, 'acc': acc}



def test_all_user(fold, rate_batch, user_batch, u_batch_id):
    training_items = []
    user_pos_test = []
    test_items = []
    whole_items = []
    #user u's items in the training set
    for i, u in enumerate(user_batch):
        train_item = []
        try:
            train_item = data_generator[fold].train_items[u]
            training_items.extend([k+i*585 for k in train_item])
        except Exception:
            training_items = []
        test_item = data_generator[fold].test_set[u]
        user_pos_test.extend([k+i*585 for k in test_item])
        all_items = set(range(ITEM_NUM))
        test_items = list(all_items - set(train_item))
        whole_items.extend([k+i*585 for k in list(all_items)])

    rate_batch_np = rate_batch.numpy()
    rate_batch_list = []
    for rate in rate_batch_np:
        rate_batch_list.extend(rate)
    #user u's items in the test set
    # for i in range(len(user_batch)):
    if args.test_flag == 'part':
        # 合并一下test_items user_pos_test rate_batch
        r, auc, aupr, acc = ranklist_by_heapq2(user_pos_test, whole_items, rate_batch_list, Ks, u_batch_id, fold)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, whole_items, rate_batch_list, Ks)
    return get_performance(user_pos_test, r, auc, aupr, acc, Ks)


def pred_one_user(rate_batch, user_batch):
    for i in range(len(user_batch)):
        # user u's ratings for user u
        rating = rate_batch[i]
        # uid
        u = user_batch[i]
        # user u's items in the training set
        try:
            predtion_items = data_generator[0].pred_items[u]
        except Exception:
            predtion_items = []
        # user u's items in the test set

        all_items = set(range(ITEM_NUM))
        pred_items = list(all_items - set(predtion_items))
        if args.test_flag == 'part':
            K_max_item_score = ranklist_by_heapq3(u, pred_items, rating, Ks)
        else:
            K_max_item_score = False

    return K_max_item_score




def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator[0].train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator[0].test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
    aupr = []
    acc = []
    return get_performance(user_pos_test, r, auc, aupr, acc, Ks)


def test(fold, model, users_to_test, drop_flag=False, batch_test_flag=False, is_test=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0., 'aupr': 0.}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(1):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                  item_batch,
                                                                  [],
                                                                  drop_flag=False)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                else:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                  item_batch,
                                                                  [],
                                                                  drop_flag=True)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            # all-item test
            item_batch = range(ITEM_NUM)

            if drop_flag == False:
                pos_output, neg_output = model(user_batch, item_batch, [], drop_flag=False, is_train=False)
                rate_batch = model.rating(pos_output).detach().cpu()
            else:
                pos_output, neg_output = model(user_batch, item_batch, [], drop_flag=True, is_train=False)
                rate_batch = model.rating(pos_output).detach().cpu()

        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
        #test_all_user()
        if is_test == False:
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)
            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['auc'] += re['auc'] / n_test_users
        if is_test == True:
            result = test_all_user(fold, rate_batch, user_batch, u_batch_id)


    pool.close()
    return result


def prediction(model, users_to_pred, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}


    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    pred_users = users_to_pred
    n_pred_users = len(pred_users)
    n_user_batchs = n_pred_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = pred_users[start: end]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                  item_batch,
                                                                  [],
                                                                  drop_flag=False)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                else:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                  item_batch,
                                                                  [],
                                                                  drop_flag=True)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            # all-item test
            item_batch = range(ITEM_NUM)

            if drop_flag == False:
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                              item_batch,
                                                              [],
                                                              drop_flag=False)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
            else:
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                              item_batch,
                                                              [],
                                                              drop_flag=True)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

        batch_result = pred_one_user(rate_batch.numpy(), user_batch)

        #test_all_user(rate_batch, user_batch, u_batch_id)
    return result
