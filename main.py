import sys
import pickle
import numpy as np
import torch
import scipy
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt

use_gpu = torch.cuda.is_available()
from Sinkhorn_distance import SinkhornDistance, SinkhornDistance_one_to_multi,SinkhornDistance_another_one_to_multi,SinkhornDistance_given_cost
device = 'cuda:0' if (torch.cuda.is_available()) else 'cpu:0'
d_cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)

eplison=0.01
sinkhorn = SinkhornDistance(eps=eplison, max_iter=200, reduction=None).to(device)   # for 1 OT

sinkhorn1 =SinkhornDistance_another_one_to_multi(eps=eplison, max_iter=200, reduction=None).to(device)
sinkhorn2 = SinkhornDistance_given_cost(eps=eplison, max_iter=200, reduction=None).to(device)


def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


def QRreduction(datas):
    ndatas = torch.qr(datas.permute(0, 2, 1)).R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas

def distribution_calibration_dan(prototype,probabi, base_means, base_cov, n_lsamples,alpha=0.21,lambd=0.3,k=10):

    index = np.argsort(-probabi.numpy())
    dim=base_means[0].shape[0]
    calibrated_mean=0
    calibrated_cov=0

    proab_reshape=np.repeat(n_lsamples*probabi.numpy(),dim,axis=0).reshape(len(base_means),dim)
    calibrated_mean= (1-lambd)*np.sum(proab_reshape *np.concatenate([base_means[:]]),axis=0)+lambd*prototype
    #
    proab_reshape_conv=np.repeat(n_lsamples*probabi.numpy(),dim*dim,axis=0).reshape(len(base_means),dim,dim)
    calibrated_cov=np.sum(proab_reshape_conv*np.concatenate([base_cov[:]]),axis=0)+alpha
    return calibrated_mean, calibrated_cov


if __name__ == '__main__':
    # ---- data loading
    dataset = 'cifar' #miniImagenet CUB cifar
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 1
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    cost_adapt = True #True False

    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas,novel_class,dataset_mean = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs, n_samples)
    # ---- Tukey's transform

    if dataset == 'miniImagenet':
        beta = 0.5
        alpha = 0.21
        lambd = 0.3
    elif dataset=='cifar':
        beta = 0.8
        alpha = 0.21
        lambd = 0.3
    else:
        beta=1
        alpha = 0.21
        lambd = 0.3

    if beta!=0:
        ndatas= torch.pow(ndatas+1e-6, beta)
    else:
        ndatas = torch.log(ndatas + 1e-6)
    # ---- Base class statistics
    base_means = []
    base_cov = []
    feature_all=[]
    shape_all=[]
    base_label = []
    base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/last/base_features.plk"%dataset
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        label = 0
        for key in data.keys():
            feature = np.array(data[key])
            if key == 0:
                feature_all_conc = feature
            else:
                feature_all_conc = np.concatenate((feature_all_conc, feature), axis=0)
            feature_all.append(feature)
            shape_all.append(feature.shape[0])
            mean = np.mean(feature, axis=0)
            base_means.append(mean)
            cov = np.cov(feature.T)
            base_cov.append(cov)
            base_label.extend([label] * len(feature))
            label =label+1

    classifier_base = LogisticRegression(max_iter=1000).fit(X=feature_all_conc, y=base_label)
    feature_all_reshape=np.zeros(shape=[len(base_means),max(shape_all),feature.shape[1]])
    sample_probabi=torch.zeros([len(base_means),max(shape_all) ])

    start_index = int(0)
    base_means_cal =[]
    base_cov_cal = []
    for ii in range(len(feature_all)):
        if shape_all[ii]==max(shape_all):
            feature_all_reshape[ii,:,:] = feature_all[ii]
        else:
            pad = max(shape_all)-shape_all[ii]
            Tmp = feature_all[ii]
            Tmp1 =np.concatenate( [Tmp, Tmp[0:pad]] )
            feature_all_reshape[ii,:,:] = Tmp1
        predict_prob = classifier_base.predict_proba(feature_all_reshape[ii,:,:])
        sample_probabi[ii,:] = torch.nn.functional.softmax(torch.from_numpy(predict_prob[:, ii]/0.3), dim=0)

    feature_torch = torch.from_numpy(feature_all_reshape).to(device)
    # ---- classification for each task
    acc_list = []
    cost_list = []

    print('Start classification for %d tasks...'%(n_runs))
    for i in tqdm(range(n_runs)):
        support_data = ndatas[i][:n_lsamples].numpy()
        support_label = labels[i][:n_lsamples].numpy()
        query_data = ndatas[i][n_lsamples:].numpy()
        query_label = labels[i][n_lsamples:].numpy()
        # ---- distribution calibration and feature sampling
        sampled_data = []
        sampled_label = []
        num_sampled = int(750/n_shot)
        base_torch=torch.from_numpy(np.concatenate([base_means[:]])).to(device)
        support_torch = torch.from_numpy(support_data).to(device)
        support_each = torch.from_numpy(support_data).to(device).unsqueeze(0)

        if cost_adapt is True:
            cost_inner, pi, C = sinkhorn1(feature_torch, support_each, sample_probabi)
            cost, Pi, C = sinkhorn2(base_torch, support_torch, cost_inner)

        else:
            cost, Pi, C = sinkhorn(base_torch, support_torch)
        ##################

        cost_list.append(cost.numpy())
        query_data_ = torch.from_numpy((query_data)).unsqueeze(-3)
        for num in range(n_lsamples):
            k=len(base_means)
            mean, cov = distribution_calibration_dan(support_data[num], Pi[:,num],base_means, base_cov, n_lsamples,alpha,lambd,k)
            sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
            sampled_label.extend([support_label[num]]*num_sampled)
        sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
        X_aug = np.concatenate([support_data, sampled_data])
        Y_aug = np.concatenate([support_label, sampled_label])
        # ---- train classifier
        classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)
        predicts = classifier.predict(query_data)
        acc = np.mean(predicts == query_label)
        acc_list.append(acc)
        if i%10==0:
            print('%s %d way %d shot  ACC : %f'%(dataset,n_ways,n_shot,float(np.mean(acc_list))))
            print(float(np.mean(cost_list)))
