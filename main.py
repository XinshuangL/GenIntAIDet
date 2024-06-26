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

from detector import Detector
import argparse
import json
import torch.nn as nn
import os
from sklearn.metrics import accuracy_score

# Settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fake_type', type=str)
opt = parser.parse_args()
exp_name = opt.fake_type
data_root = r'generated_data/'

# Load train data
def load_epoch(data_folder, epoch):
    with open(f'{data_folder}/epoch_{epoch}.json', 'r') as f:
        json_dict = json.load(f)
        step_num = len(json_dict)
        for i in range(step_num):
            for k, v in json_dict[i].items():
                json_dict[i][k] = torch.tensor(v).cuda()
    return json_dict

# Load feature ranks
with open(f'feature_rank_{opt.fake_type}_2x2.json', 'r') as f:
    pair_list = json.load(f)
feature_ids = [item[0] for item in pair_list]

# Prepare for the model and optimizer
model = Detector(feature_ids, feature_rate=0.9).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

# Train or load the checkpoint
if not os.path.exists(f'checkpoints/{exp_name}.pth'):
    print('start train, becasuse there is no existing', exp_name)
    os.makedirs(f'checkpoints/', exist_ok=True)
    for epoch in range(100):
        epoch_data = load_epoch(data_root + opt.fake_type + '_data_2x2', epoch)
        for step_data in epoch_data:
            input = step_data['feature']
            label = step_data['label'].float()
            output = model(input)
            
            loss = loss_fn(output, label) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'epoch: {epoch}, loss: {epoch}')

    torch.save(model.state_dict(), f'checkpoints/{exp_name}.pth')
else:
    model.load_state_dict(torch.load(f'checkpoints/{exp_name}.pth', map_location='cuda'))
    
# evaluation metric
def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc    

# load test data
model.eval()
os.makedirs(f'test/', exist_ok=True)
test_type = 'style_transfer' if opt.fake_type == 'inpainting' else 'inpainting'
with open(f'{data_root}/{test_type}_data_2x2_test.json', 'r') as f:
    json_dict = json.load(f)

# prediction
y_true, y_pred_levels, index_levels = [], [[], [], []], [[], [], []]
with torch.no_grad():
    for step_data in json_dict:
        label = step_data['label']
        y_true.append(float(label[0]))

        # hierarchical AI-modified image detection process
        feature_tree = step_data['feature_tree']
        for level in range(3):
            output, index = model(torch.tensor(feature_tree['feature']).cuda(), True)
            y_pred_levels[level].extend(output.sigmoid().flatten().tolist())
            str_key = f'{index[0]}_{index[1]}'
            index_levels[level].append(str_key)
            if 'sub_features' in feature_tree.keys() and not feature_tree['sub_features'] == None:
                feature_tree = feature_tree['sub_features'][str_key]
y_true, y_pred_levels = np.array(y_true), np.array(y_pred_levels)
y_pred_levels[y_pred_levels > 0.5] = 1
y_pred_levels[y_pred_levels < 0.51] = 0

# evaluate the predictions
results = {}
for level in range(1, 3):
    y_pred_levels[level] = 1 - (1 - y_pred_levels[level]) * (1 - y_pred_levels[level - 1])
for level in range(3):
    r_acc, f_acc, acc = calculate_acc(y_true, y_pred_levels[level], 0.5)
    result = {
        'acc': acc,
        'r_acc': r_acc,
        'f_acc': f_acc,
        'indexes': index_levels[level]
    }
    results[level] = result

# save results
with open(f'test/{exp_name}.json', 'w') as f:
    json.dump(results, f, indent=2)
