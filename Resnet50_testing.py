"""

Le code présenté ici est la propriété de Amiel Allab et du Laboratoire de Recherche en Imagerie et Orthopédie LIO-ÉTS.
Il est exclusivement destiné à des fins de recrutement et ne doit en aucun cas être utilisé à d'autres fins sans autorisation préalable.

Toute utilisation, reproduction, ou diffusion du code en dehors de ce cadre est strictement interdite.
En accédant à ce code, vous acceptez de respecter ces conditions et de ne pas l'utiliser à des fins personnelles, commerciales
ou de recherche sans le consentement explicite du laboratoire LIO-ÉTS.

"""

from __future__ import print_function, division
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import os
import ConfusionMatrixMetrics as CMM
import ConfusionMatrixCalculator as CMC

plt.ion()   # interactive mode

# Vérification de l'existence du répertoire et des transformations
def check_directory_exists(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Le répertoire {directory} n'existe pas.")
    return directory

# Vérification si un modèle existe avant de le charger
def check_model_exists(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_path} n'a pas été trouvé.")
    return model_path

test_transforms = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Spécifiez le chemin vers le répertoire d'images test
test_dir = 'C:/Git/Classification_renet50/DATA/'
try:
    test_dir = check_directory_exists(test_dir)
except FileNotFoundError as e:
    print(e)
    exit(1)
n_classes = 2

# Chargement des données
try:
    testset = ImageFolder(test_dir, transform=test_transforms)
except Exception as e:
    print(f"Erreur lors du chargement du jeu de données : {e}")
    raise

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)
testset_size = len(testset)

# Chargement du modèle et vérification
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    model_path = 'Resnet50_modele.pt'
    model_path = check_model_exists(model_path)
    model = torch.load(model_path)
    model.eval()
except FileNotFoundError as e:
    print(e)
    raise
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    raise

model = model.to(device)
criterion = nn.CrossEntropyLoss()

# Appliquer le modèle sur le jeu de test
inputs, classes = next(iter(testloader))
running_loss = 0.0
running_corrects = 0
k = 0

# Initialisation de la classe avec 2 classes
cm_calculator = CMC.ConfusionMatrixCalculator(n_classes=n_classes)

# Itération sur les données de test
for inputs, labels in testloader:
    try:
        # Vérification des données et du modèle sur le bon appareil
        if inputs.device != device:
            inputs = inputs.to(device)
        if labels.device != device:
            labels = labels.to(device)

        outputs = model(inputs)
        outputs_prob = torch.nn.functional.softmax(outputs, dim=1)
        # Mise à jour de la matrice de confusion
        conf_matrix = cm_calculator.update_confusion_matrix(outputs_prob, labels)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
    except Exception as e:
        print(f"Erreur durant l'itération de test : {e}")
        continue

    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

test_loss = running_loss / testset_size
test_acc = running_corrects.double() / testset_size

print(f'Test loss: {test_loss:.4f}')
print(f'Test Acc: {test_acc:.4f}')
print(conf_matrix)
metriques = CMM.ConfusionMatrixMetrics(conf_matrix=conf_matrix, n_classes=n_classes)
for c in range(n_classes):
    TP, TN, FP, FN, sensitivity, specificity = metriques.calcul_indc_matrice_confu(c)
    print(f'Class {c}\nTP {TP}, TN {TN}, FP {FP}, FN {FN}')
    print(f'Sensitivity = {sensitivity:.4f}')
    print(f'Specificity = {specificity:.4f}')
