import torch
import os

from .model import vit_b_16_model
from timeit import default_timer as timer
from typing import Tuple, Dict

brain_xray_model, brain_xray_transforms = vit_b_16_model()

class_names = ["glioma_tumor", "meningioma_tumor", "normal", "pituitary_tumor"]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pretrained_vit_b_16_brain_xray.pth")

brain_xray_model.load_state_dict(
    torch.load(
        f = MODEL_PATH,
        map_location = torch.device("cpu")
    )
)

def predict_image(img) -> Tuple[Dict, float]:
  start_time = timer()

  img = brain_xray_transforms(img).unsqueeze(dim = 0)
  brain_xray_model.eval()
  with torch.inference_mode():
      y_logit = brain_xray_model(img)
      y_prob = torch.softmax(y_logit, dim = 1)

  pred_labels_and_probs = {class_names[i] : round(float(y_prob[0][i]), 4) for i in range(len(class_names))}
  end_time = timer()
  pred_time = round(end_time - start_time, 5)

  return pred_labels_and_probs, pred_time