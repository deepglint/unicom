The example below uses scikit-learn to perform logistic regression on image features.

```python
import os
import unicom
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, StanfordCars
from tqdm import tqdm

# Load the model
model, preprocess = unicom.load("ViT-L/14")
model = model.cuda()

root = os.path.expanduser("~/.cache")
train = StanfordCars(root, download=True, split="train", transform=preprocess)
test = StanfordCars(root, download=True, split="test", transform=preprocess)


def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model(images.cuda())

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)
classifier = LogisticRegression(random_state=0, C=0.601, max_iter=1000, verbose=0)
classifier.fit(train_features, train_labels)
# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")

```