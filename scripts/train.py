import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
import torchvision.transforms as transforms
import datasets
import os
from datasets import load_dataset


def evaluate_model(model, dataset, feature_extractor):
    model.eval()
    correct = 0
    total = 0
    for example in dataset:
        inputs = feature_extractor(example["image"], return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            outputs = model(inputs)
            predicted = torch.argmax(outputs.logits, dim=1).item()
        if predicted == example["label"]:
            correct += 1
        total += 1
    accuracy = 100 * correct / total
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy


def inference(model, image_path, feature_extractor):
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(image, return_tensors="pt")["pixel_values"]
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        predicted = torch.argmax(outputs.logits, dim=1).item()
    print(f"Predicted class: {predicted}")
    return predicted


def main():
    dataset = load_dataset("imagefolder", data_dir="data/processed/splits")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    def transform(example):
        example["pixel_values"] = feature_extractor(example["image"], return_tensors="pt")["pixel_values"][0]
        return example

    dataset = dataset.with_transform(transform)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(dataset["train"].features["label"].names)
    )

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
    print("Model saved successfully!")

    print("Evaluating model...")
    evaluate_model(model, dataset["test"], feature_extractor)


if __name__ == '__main__':
    main()
