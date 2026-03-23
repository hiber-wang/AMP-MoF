import time
import torch
import torch.nn as nn
import sys
sys.path.append("..")
from TimeSformer.timesformer.models.vit import TimeSformer
from dataset import DrivingCaptures
from .autoencoder import Encoder
from torchvision import models

class AbnormalDetector(nn.Module):
    def __init__(self, img_size, num_frames, embedding_dim):
        super(AbnormalDetector, self).__init__()
        print("num_frames:", num_frames)
        if num_frames == 1:
            self.video_feature_extractor = models.resnet101(pretrained=True)
            for parma in self.video_feature_extractor.parameters():
                    parma.requires_grad = False
            num_ftrs = self.video_feature_extractor.fc.in_features
            self.video_feature_extractor.fc = nn.Linear(in_features=num_ftrs, out_features=num_ftrs, bias=False)
            torch.nn.init.eye_(self.video_feature_extractor.fc.weight)
        else:
            if num_frames == 8 or num_frames == -8:
                num_frames = 8
                timesformer_pretrain_model = "../pretrained_models/TimeSformer_divST_8x32_224_K400.pyth"
            elif num_frames == 16:
                timesformer_pretrain_model = "../pretrained_models/TimeSformer_divST_16x16_448_K400.pyth"
            self.video_feature_extractor = TimeSformer(img_size=img_size, num_classes=400, num_frames=num_frames, pretrained_model=timesformer_pretrain_model)
            num_ftrs = self.video_feature_extractor.model.head.in_features
            self.video_feature_extractor.model.head = torch.nn.Linear(num_ftrs, num_ftrs, bias=False)
            torch.nn.init.eye_(self.video_feature_extractor.model.head.weight)
            for param in self.video_feature_extractor.parameters():
                param.requires_grad = False
        
        self.encoder = Encoder(num_ftrs, embedding_dim)

    def forward(self, front, is_only_extract=False):
        out = self.video_feature_extractor(front)
        if not is_only_extract:
            out = self.encoder(out)
        return out    


if __name__ == "__main__":
    x = torch.randn([1, 3, 8, 224, 224]).cuda()
    ad = AbnormalDetector(224, 8, 225).cuda()
    out = ad(x)
    print(out.shape)

