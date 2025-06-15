import torch
import torchvision

def vit_b_16_model(classes : int = 4):
  weights = torchvision.models.ViT_B_16_Weights.DEFAULT
  transform = weights.transforms()
  model = torchvision.models.vit_b_16(weights = weights)

  for param in model.parameters():
    param.requires_grad = False

  model.heads = torch.nn.Sequential(
      torch.nn.Linear(in_features = 768, out_features = classes)
  )
  return model, transform