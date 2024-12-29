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
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from TrainModele import ModelTrainer

plt.ion()   # interactive mode


# Fonction pour vérifier les erreurs de chemin
def check_path_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le chemin spécifié n'existe pas : {path}")

# Fonction pour vérifier si le dataset est valide
def check_dataset(dataset):
    if len(dataset) == 0:
        raise ValueError("Le dataset est vide. Vérifiez le répertoire et les fichiers.")

# Fonction pour retourner la somme des bonnes classification
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

# afficher courbe d'entrainement et faire pause 2s
def affichercourbe1(x, y):
    plt.xlabel('Iterations')
    plt.ylabel('Erreur d apprentissage')
    plt.plot(x, y)
    plt.pause(2)  # pause a bit so that plots are updated
    plt.savefig('Train_Val_loss_resnet50.png')
# afficher courbe de validation et sauvegarde de la figure
def affichercourbe2(x, y):
    plt.xlabel('Iterations')
    plt.ylabel('Erreur d apprentissage')
    plt.plot(x, y)
    plt.pause(2)  # pause a bit so that plots are updated
    plt.savefig('Train_Val_loss_resnet50.png')


# Pretraitement des images d'entrainement ('train') et des images de validation ('val')
# Augmentation des données et normalisation pour l'entrainement
# Juste la normalisation pour la validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# Fonction pour charger les données d'entrainement et de validation avec un tirage aléatoire du meme dossier
# Pourcentage des données utilisées en validation a définir dans val_split
def train_val_dataset(dataset, val_split=0.20):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

################################################################################
# Répertoire de données d entrainement et de validation
data_dir = 'C:/Git/Classification_renet50/DATA/'
# Vérification du chemins
try:
    check_path_exists(data_dir)
except FileNotFoundError as e:
    print(e)
    exit(1)

# Chargement du dataset
try:
    dataset = ImageFolder(data_dir, transform=data_transforms['train'])
    check_dataset(dataset)
except Exception as e:
    print(f"Erreur lors du chargement du dataset: {e}")
    exit(1)

print(len(dataset))

datasets = train_val_dataset(dataset)
print(len(datasets['train']))
print(len(datasets['val']))

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
class_names = dataset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get a batch of training data
#inputs, classes = next(iter(dataloaders['train']))

# chargement du modele resnet50 pré-entrainé sur ImageNet
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)

# Fixer les hyperparametres de la fonction d'entrainement
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Entraînement du modèle avec gestion des exceptions
try:
    # Entraînement du modèle
    trainer = ModelTrainer(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10, dataloaders=dataloaders,
                           device=device)
    model, loss_train, loss_val = trainer.train()

    # Affichage des pertes et précisions
    print(f"Training Loss: {loss_train}")
    print(f"Validation Loss: {loss_val}")

except Exception as e:
    print(f"Erreur lors de l'entraînement du modèle: {e}")
    exit(1)

# Affichage des courbes d entrainement et de validation
affichercourbe1(loss_train[0, :], loss_train[1, :])
affichercourbe2(loss_val[0, :], loss_val[1, :])

# Sauvegarde du modèle
torch.save(model_ft, 'Resnet50_modele.pt')
