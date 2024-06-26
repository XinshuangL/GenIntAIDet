from .clip import clip 
import torch.nn as nn
import torch

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( 768, num_classes )
        print('use 2x2 clip model')

    def forward(self, x, return_feature=False):
        N, C, H, W = x.shape
        H2 = int(H * 2 / 3 + 0.5)
        W2 = int(W * 2 / 3 + 0.5)
        H1 = int(H / 3 + 0.5)
        W1 = int(H / 3 + 0.5)

        x00 = x[:, :, :H2, :W2]
        x01 = x[:, :, :H2, W1:]
        x10 = x[:, :, H1:, :W2]
        x11 = x[:, :, H1:, W1:]

        features00 = self.model.encode_image(x00).view(N, 768, 1, 1)
        features01 = self.model.encode_image(x01).view(N, 768, 1, 1)
        features10 = self.model.encode_image(x10).view(N, 768, 1, 1)
        features11 = self.model.encode_image(x11).view(N, 768, 1, 1)

        features = torch.cat([
            torch.cat([features00, features10], dim=2),
            torch.cat([features01, features11], dim=2)
        ], dim=3)

        if return_feature:
            return features
        return self.fc(features)

