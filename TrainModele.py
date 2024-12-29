import time
import copy
import torch
import numpy as np

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, num_epochs, dataloaders, device):
        """
        Initialise la classe ModelTrainer.

        :param model: Le modèle de réseau de neurones à entraîner.
        :param criterion: La fonction de perte.
        :param optimizer: L'optimiseur pour le modèle.
        :param scheduler: Le planificateur de taux d'apprentissage.
        :param num_epochs: Le nombre d'époques pour l'entraînement.
        :param dataloaders: Le dictionnaire des dataloaders ('train' et 'val').
        :param device: Le périphérique sur lequel entraîner le modèle (e.g., 'cpu' ou 'cuda').
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.dataloaders = dataloaders
        self.device = device
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0
        self.loss_train = np.zeros((2, num_epochs), float)
        self.loss_val = np.zeros((2, num_epochs), float)

    def train(self):
        """
        Entraîne le modèle en utilisant les données fournies.

        Retourne le modèle entraîné et les valeurs des pertes d'entraînement et de validation.
        """
        since = time.time()

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)

            # Chaque époque a une phase d'entraînement et de validation
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Met le modèle en mode entraînement
                else:
                    self.model.eval()   # Met le modèle en mode évaluation

                running_loss = 0.0
                running_corrects = 0

                # Itérer sur les données.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Zéro les gradients des paramètres
                    self.optimizer.zero_grad()

                    # Propagation avant
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Statistiques
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                if phase == 'train':
                    self.loss_train[0, epoch] = epoch
                    self.loss_train[1, epoch] = epoch_loss
                if phase == 'val':
                    self.loss_val[0, epoch] = epoch
                    self.loss_val[1, epoch] = epoch_loss
                if phase == 'val' and epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
        print(f'Best val Acc: {self.best_acc:.4f}')

        self.model.load_state_dict(self.best_model_wts)
        return self.model, self.loss_train, self.loss_val
