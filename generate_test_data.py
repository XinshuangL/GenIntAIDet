import os
import json
from tqdm import tqdm
from data import create_dataloader_test
from options.options import Options
from clip_models import get_model

def pred_tree(img_tree, model):
    img = img_tree['img']
    feature = model(img.cuda(), return_feature=True).cpu().tolist()
    if 'sub_imgs' in img_tree.keys():
        sub_features = {}
        for k, v in img_tree['sub_imgs'].items(): 
            sub_features[k] = pred_tree(v, model)
        return {
            'feature': feature,
            'sub_features': sub_features
        }
    else:
        return{
            'feature': feature,
            'sub_features': None
        }

if __name__ == '__main__':
    opt = Options().parse()
    model = get_model(opt.arch)
    data_loader = create_dataloader_test(opt)

    step_list = []

    print(f'generate test data for {opt.fake_type}')
    for i, data in tqdm(enumerate(data_loader)):
        img_tree = data[0]
        label = data[1].cuda()

        feature_tree = pred_tree(img_tree, model)
        step_list.append({
            'feature_tree': feature_tree,
            'label': label.tolist()
        })

    os.makedirs('generated_data', exist_ok=True)
    with open(f'generated_data/{opt.fake_type}_data_2x2_test.json', 'w') as f:
        json.dump(step_list, f, indent=2)
