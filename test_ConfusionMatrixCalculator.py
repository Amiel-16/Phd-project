"""

Le code présenté ici est la propriété de Amiel Allab et du Laboratoire de Recherche en Imagerie et Orthopédie LIO-ÉTS.
Il est exclusivement destiné à des fins de recrutement et ne doit en aucun cas être utilisé à d'autres fins sans autorisation préalable.

Toute utilisation, reproduction, ou diffusion du code en dehors de ce cadre est strictement interdite.
En accédant à ce code, vous acceptez de respecter ces conditions et de ne pas l'utiliser à des fins personnelles, commerciales
ou de recherche sans le consentement explicite du laboratoire LIO-ÉTS.

"""

import unittest
import torch
from ConfusionMatrixCalculator import ConfusionMatrixCalculator

class TestConfusionMatrixCalculator(unittest.TestCase):

    # Test de la fonction confusion_matrix TP = 1 et TN = 1
    def test_positive_update_confusion_matrix(self):
        # Exemple de prédictions et de véritables étiquettes
        preds = torch.tensor([[0.2, 0.8], [0.7, 0.3]])  # Probabilités
        labels = torch.tensor([1, 0])  # Classes réelles
        cm_calculator = ConfusionMatrixCalculator(n_classes=2)
        conf_matrix = cm_calculator.update_confusion_matrix(preds, labels)
        self.assertEqual(conf_matrix[0, 0], 1)  # TN = 1
        self.assertEqual(conf_matrix[1, 1], 1)  # TP = 1
        self.assertEqual(conf_matrix[0, 1], 0)  # FN = 0
        self.assertEqual(conf_matrix[1, 0], 0)  # FP = 0

    def test_false_update_confusion_matrix(self):
        # Exemple de prédictions et de véritables étiquettes
        preds = torch.tensor([[0.3, 0.7], [0.9, 0.1]])  # Probabilités
        labels = torch.tensor([0, 1])  # Classes réelles
        cm_calculator = ConfusionMatrixCalculator(n_classes=2)
        conf_matrix = cm_calculator.update_confusion_matrix(preds, labels)
        # Test de la mise à jour de la matrice de confusion
        self.assertEqual(conf_matrix[0, 0], 0)  # TN = 0
        self.assertEqual(conf_matrix[1, 1], 0)  # TP = 0
        self.assertEqual(conf_matrix[0, 1], 1)  # FN = 1
        self.assertEqual(conf_matrix[1, 0], 1)  # FP = 1

    def test_invalid_input_size(self):
        preds = torch.tensor([[0.1, 0.9], [0.7, 0.3]])
        labels = torch.tensor([1, 0, 0])  # Taille incompatible
        cm_calculator = ConfusionMatrixCalculator(n_classes=2)
        with self.assertRaises(ValueError):
            cm_calculator.update_confusion_matrix(preds, labels)

    def test_invalid_type(self):
        preds = [[0.1, 0.9], [0.7, 0.3]]  # Pas un tensor
        labels = torch.tensor([1, 0, 0])
        cm_calculator = ConfusionMatrixCalculator(n_classes=2)
        with self.assertRaises(TypeError):
            cm_calculator.update_confusion_matrix(preds, labels)

if __name__ == '__main__':
    unittest.main()
