import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
import torchvision.transforms as transforms
import datasets
import os
from datasets import load_dataset


# Load dataset from folder
dataset = load_dataset("imagefolder", data_dir="/content/drive/MyDrive/splits")

# Define preprocessing
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
def transform(example):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])
    example["pixel_values"] = [transform(img.convert("RGB")) for img in example["image"]]
    return example

dataset = dataset.map(transform, batched=True)
dataset = dataset.remove_columns(["image"]).rename_column("label", "labels")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
num_labels = len(dataset["train"].features["labels"].names)
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=num_labels)
model.to(device)

training_args = TrainingArguments(
        output_dir="models/outputs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        save_total_limit=2,
        logging_dir="logs",
        logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=feature_extractor,
)

trainer.train()
model.save_pretrained("models/outputs/vit_finetuned")

def evaluate():
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)
    return metrics

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])
    pixel_values = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    label_names = dataset["train"].features["labels"].names
    return label_names[predicted_class]

evaluate()
