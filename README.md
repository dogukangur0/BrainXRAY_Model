Demo: 👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/dogukang/BrainXRAY)

🧠 BrainXRAY - Image Classification with EffNet_B0 and ViT_B_16

## 📦 Teknologies
- 🔍 Model: EfficientNet_B0 and ViT_B_16 (`torchvision`)
- 📚 Dataset: Brain XRAY (4 class (GLIOMA - MENINGIOMA - NORMAL - PITUITARY)) [Dataset](https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256)
- 🧠 Training: PyTorch, torchvision
- 🖼️ UI: Gradio
- 📁 Weights: Stored in `.pth`, auto-loaded on inference

## 🚀 Train Summary
- Epochs: 10
- Loss Function: `nn.CrossEntropyLoss()`
- Optimizer: `Adam`

📌 Notes:
The model was loaded using torchvision.models.efficientnet_b0(weights=...) and vit_b_16(weights=...).

Results:
EffNet_B0 : Epoch: 10 | train_loss: 0.5355 | train_acc: 78.8670 | test_loss: 0.6302 | test_acc: 77.6864
ViT_B_16  : Epoch: 10 | train_loss: 0.3275 | train_acc: 89.0126 | test_loss: 0.4827 | test_acc: 82.6754

![Sample Prediction](xray_images.PNG)

___________________________________________________________________________________________________________________________________________________________________

FastAPI:

Demo on local computer with FastAPI

![Sample_Prediction](xray_images_fastapi.PNG)


