import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import open_clip

class CLIP_Encoder(nn.Module):
    def __init__(self, clip_weights, size=224):
        super(CLIP_Encoder, self).__init__()
        self.backbone_clip = clip_weights
        self.size = size
        self.embed_dim = self.backbone_clip.output_dim

    def forward(self, x):
        x = transforms.functional.center_crop(x, [self.size, self.size])
        x = transforms.functional.normalize(x, mean=[0.48145466, 0.4578275, 0.40821073],
                                            std=[0.26862954, 0.26130258, 0.27577711])
        x = self.backbone_clip(x)
        x = x.to(dtype=torch.float32)
        return x

    def forward_single(self, x):
        return self.forward(x)

    def embed(self, image):
        image = torch.unsqueeze(image, 0)
        image = self.forward(image)
        image = torch.squeeze(image, 0)
        return image

def get_clip(model_version, clip_dataset_and_epoch, size=224):
    #assert model_version in ["ViT-L-14", "ViT-B-16", "ViT-H-14", "ViT-g-14"]
    #assert clip_dataset_and_epoch in ["laion400m_e31", "laion2b_s32b_b79k", "laion2b_s12b_b42k"]
    print("Loading CLIP {} - {}...".format(model_version, clip_dataset_and_epoch))
    # load clip vit-l-14 laion400m-e31
    clip_backbone = open_clip.create_model(model_version, clip_dataset_and_epoch).visual.eval()
    clip_encoder = CLIP_Encoder(clip_backbone, size)
    print("Success.")
    return clip_encoder

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    clip_encoder = get_clip("ViT-L-14", "laion400m_e31", 224)
    clip_encoder.to(device).eval()
    test_tensor = torch.rand((1,3, 350,350))
    output_tensor = clip_encoder(test_tensor.to(device))
    print(output_tensor.shape)