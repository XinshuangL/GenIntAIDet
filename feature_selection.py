import torch
import numpy as np
import random
SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
set_seed()

import argparse
import json

data_root = r'generated_data/'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fake_type', type=str)

opt = parser.parse_args()

def load_epoch(data_folder, epoch):
    with open(f'{data_folder}/epoch_{epoch}.json', 'r') as f:
        json_dict = json.load(f)
        step_num = len(json_dict)
        for i in range(step_num):
            for k, v in json_dict[i].items():
                json_dict[i][k] = torch.tensor(v)
    return json_dict

real_result = {
    'mu': 0,
    'sigma': 0,
    'count': 0
}
fake_result = {
    'mu': 0,
    'sigma': 0,
    'count': 0
}

for epoch in range(100):
    epoch_data = load_epoch(data_root + opt.fake_type + '_data_2x2', epoch)
    for step_data in epoch_data:
        input = step_data['feature']
        label = step_data['label'].float()
        real_input = input[label < 0.5]
        fake_input = input[label >= 0.5]
        N_real = real_input.shape[0]
        N_fake = fake_input.shape[0]
        real_result['count'] += N_real
        fake_result['count'] += N_fake

        real_result['mu'] += real_input.mean((0,2,3)) * N_real
        fake_result['mu'] += fake_input.mean((0,2,3)) * N_fake

real_result['mu'] /= real_result['count']
fake_result['mu'] /= fake_result['count']

real_result['count'] = 0
fake_result['count'] = 0
for epoch in range(100):
    epoch_data = load_epoch(data_root + opt.fake_type + '_data_2x2', epoch)
    for step_data in epoch_data:
        input = step_data['feature']
        label = step_data['label'].float()
        real_input = input[label < 0.5]
        fake_input = input[label >= 0.5]
        N_real = real_input.shape[0]
        N_fake = fake_input.shape[0]
        real_result['count'] += N_real
        fake_result['count'] += N_fake

        real_dif = real_input - real_result['mu'].view(1, 768, 1, 1)
        fake_dif = fake_input - fake_result['mu'].view(1, 768, 1, 1)

        real_result['sigma'] += (real_dif * real_dif).mean((0,2,3)) * N_real
        fake_result['sigma'] += (fake_dif * fake_dif).mean((0,2,3)) * N_fake

real_result['sigma'] /= (real_result['count'] - 1)
fake_result['sigma'] /= (fake_result['count'] - 1)

real_mu, real_sigma = real_result['mu'], real_result['sigma']
fake_mu, fake_sigma = fake_result['mu'], fake_result['sigma']

d = real_mu.shape[0]

def feature_selection(real_mu, real_sigma, fake_mu, fake_sigma):
    delta_mu = real_mu - fake_mu
    mu_score = delta_mu * delta_mu
    real_sigma_list = (real_sigma * torch.diag(torch.ones((d)))).sum(dim=1)
    fake_sigma_list = (fake_sigma * torch.diag(torch.ones((d)))).sum(dim=1)
    sigma_score = real_sigma_list + fake_sigma_list - 2 * (real_sigma_list * fake_sigma_list).sqrt()
    score = mu_score + sigma_score
    pair_list = []
    for i in range(d):
        pair_list.append([i, float(score[i])])
    pair_list.sort(key=lambda item: item[1], reverse=True)
    return pair_list

pair_list = feature_selection(real_mu, real_sigma, fake_mu, fake_sigma)
with open(f'feature_rank_{opt.fake_type}_2x2.json', 'w') as f:
    json.dump(pair_list, f, indent=2)

