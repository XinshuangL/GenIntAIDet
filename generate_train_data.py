import os
import json
from tqdm import tqdm
from data import create_dataloader
from options.options import Options
from clip_models import get_model

if __name__ == '__main__':
    opt = Options().parse()
    model = get_model(opt.arch)
    data_loader = create_dataloader(opt)

    os.makedirs(f'generated_data/{opt.fake_type}_data_2x2', exist_ok=True)
    print(f'generate train data for {opt.fake_type}')
    for epoch in tqdm(range(opt.niter)):
        step_list = []
        for i, data in enumerate(data_loader):
            img = data[0].cuda()
            label = data[1].cpu()
            feature = model(img, return_feature=True).cpu()
            step_list.append({
                'feature': feature.tolist(),
                'label': label.tolist()
            })

        with open(f'generated_data/{opt.fake_type}_data_2x2/epoch_{epoch}.json', 'w') as f:
            json.dump(step_list, f, indent=2)
