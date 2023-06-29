import torch
import torch.nn as nn
import timm
from torch.nn import functional as F
from torchvision import transforms

class SiameseNet(nn.Module):
    def __init__(self, backbone_name='tf_efficientnet_b7', output_size=256):
        super(SiameseNet, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        self.backbone.reset_classifier(0)  # Remove the classification head
        self.final_pool = nn.AdaptiveAvgPool1d(output_size=output_size)
        self.embed_dim = output_size

    def forward_single(self, x):
        x = transforms.functional.normalize(x, mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        embedding = self.final_pool(self.backbone(x))
        return embedding

    def forward(self, x1, x2):
        embedding1 = self.forward_single(x1)
        embedding2 = self.forward_single(x2)
        return embedding1, embedding2

    def embed(self, image):
        image = torch.unsqueeze(image, 0)
        image = self.forward_single(image)
        image = torch.squeeze(image, 0)
        return image

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        distance = F.pairwise_distance(embedding1, embedding2)
        loss = 0.5 * (1 - label) * torch.pow(distance, 2) + 0.5 * label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return loss.mean()

if __name__ == "__main__":
    print(timm.list_models('*regnet*', pretrained=True)) # efficientnet_b0
    #print(timm.list_models('*mobil*', pretrained=True))
    siamese_model = SiameseNet()
    input_tensor = torch.rand((3,224,224))
    output_tensor = siamese_model.embed(input_tensor)
    print(output_tensor.shape)
    #print(siamese_model)
