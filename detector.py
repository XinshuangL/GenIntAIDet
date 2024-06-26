import torch.nn as nn
import torch
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, c):
        super(ChannelAttention, self).__init__()
        self.scale = nn.Sequential(
            nn.Linear(c, c, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, feature):
        scale = self.scale(feature.mean((2, 3))).unsqueeze(2).unsqueeze(3) + 0.5
        return scale * feature

class SpatialAttention(nn.Module):
    def __init__(self, c):
        super(SpatialAttention, self).__init__()        
        self.qk_transform = nn.Conv2d(c, c, 1)

    def vectorize(self, feature):
        N, C, H, W = feature.shape
        return feature.view(N, C, H * W)

    def forward(self, feature):
        # To avoid overfitting, 
        # we use the same transformation for the query and key,
        # and no transformation for the value.
        q = self.vectorize(self.qk_transform(feature))
        k = self.vectorize(self.qk_transform(feature))       
        v = self.vectorize(feature)
        attention = F.softmax(torch.bmm(q.permute(0, 2, 1), k), dim=2)
        return (feature + torch.bmm(v, attention.permute(0, 2, 1)).view(*feature.shape)) / 2

class Detector(nn.Module):
    def __init__(self, feature_ids, feature_rate=None):
        super(Detector, self).__init__()
        
        self.feature_ids = feature_ids
        self.feature_num = int(feature_rate * 768 + 0.5)
        self.feature_num_core = int(0.01 * 768 + 0.5)

        attention_layers = []
        attention_layers.append(ChannelAttention(self.feature_num_core))
        attention_layers.append(SpatialAttention(self.feature_num_core))
        self.attention_layers = nn.Sequential(*attention_layers)

        self.active_rate = 10.0

        layers = []
        layers.append(nn.Conv2d(self.feature_num + self.feature_num_core, 1, 1, padding=0))
        self.layers = nn.Sequential(*layers)

    def forward(self, feature, return_index=False):
        feature_core = feature[:, self.feature_ids[:self.feature_num_core]]
        feature_core = self.attention_layers(feature_core)
        feature = feature[:, self.feature_ids[:self.feature_num]]
        feature = torch.cat([feature, feature_core], dim=1)
    
        detection_map = self.layers(feature)

        max_score = F.max_pool2d(detection_map, 2)
        avg_score = F.avg_pool2d(detection_map, 2)
        score = (max_score * (self.active_rate - 1/4) + avg_score) / (self.active_rate + 3/4)
        
        if not return_index:
            return score.view(-1)
        else:
            H_indexes = torch.zeros_like(detection_map)
            for i in range(H_indexes.shape[2]):
                H_indexes[:,:,i,:] = i
            W_indexes = torch.zeros_like(detection_map)
            for i in range(W_indexes.shape[3]):
                W_indexes[:,:,:,i] = i

            select = (detection_map == max_score)

            H = H_indexes[select]
            W = W_indexes[select]

            return score.view(-1), (int(H + 0.5), int(W + 0.5))
