"""

Le code présenté ici est la propriété de Amiel Allab et du Laboratoire de Recherche en Imagerie et Orthopédie LIO-ÉTS.
Il est exclusivement destiné à des fins de recrutement et ne doit en aucun cas être utilisé à d'autres fins sans autorisation préalable.

Toute utilisation, reproduction, ou diffusion du code en dehors de ce cadre est strictement interdite.
En accédant à ce code, vous acceptez de respecter ces conditions et de ne pas l'utiliser à des fins personnelles, commerciales
ou de recherche sans le consentement explicite du laboratoire LIO-ÉTS.

"""

import unittest
import torch
from ConfusionMatrixMetrics import ConfusionMatrixMetrics

class TestConfusionMatrixMetrics(unittest.TestCase):
    # Test  de la fonction calcul des indicateur de la matrice de confusion
    def test_valid_confusion_matrix(self):
        conf_matrix = torch.tensor([[50, 10], [5, 35]], dtype=torch.int)
        cm_metrics = ConfusionMatrixMetrics(conf_matrix, n_classes=2)
        TP, TN, FP, FN, sensitivity, specificity = cm_metrics.calcul_indc_matrice_confu(0)
        self.assertEqual(TP, 50)
        self.assertEqual(TN, 35)
        self.assertEqual(FP, 10)
        self.assertEqual(FN, 5)
        self.assertEqual(sensitivity, 0.9090909090909091)
        self.assertEqual(specificity, 0.7777777777777778)

    def test_zero_case(self):
        conf_matrix = torch.tensor([[0, 0], [0, 0]], dtype=torch.int)
        cm_metrics = ConfusionMatrixMetrics(conf_matrix, n_classes=2)
        TP, TN, FP, FN, sensitivity, specificity = cm_metrics.calcul_indc_matrice_confu(0)
        self.assertEqual(TP, 0)
        self.assertEqual(TN, 0)
        self.assertEqual(FP, 0)
        self.assertEqual(FN, 0)
        self.assertEqual(sensitivity, 0)
        self.assertEqual(specificity, 0)

    def test_invalid_class_index(self):
        conf_matrix = torch.tensor([[50, 10], [5, 35]], dtype=torch.int)
        cm_metrics = ConfusionMatrixMetrics(conf_matrix, n_classes=2)
        with self.assertRaises(IndexError):
            cm_metrics.calcul_indc_matrice_confu(0)

    def test_invalid_conf_matrix_shape(self):
        conf_matrix = torch.tensor([50, 10, 5, 35], dtype=torch.int)  # Non carrée
        cm_metrics = ConfusionMatrixMetrics(conf_matrix, n_classes=2)
        with self.assertRaises(ValueError):
            cm_metrics.calcul_indc_matrice_confu(0)

    def test_non_tensor_conf_matrix(self):
        conf_matrix = [[50, 10], [5, 35]]  # Pas un tensor
        cm_metrics = ConfusionMatrixMetrics(conf_matrix, n_classes=2)
        with self.assertRaises(TypeError):
            cm_metrics.calcul_indc_matrice_confu(0)

    def test_invalid_class_index(self):
        """
        Teste que l'on lève une exception si l'indice de la classe est invalide.
        """
        conf_matrix = torch.tensor([[5, 1], [2, 7]])
        cm_metrics = ConfusionMatrixMetrics(conf_matrix, n_classes=2)

        with self.assertRaises(IndexError):
            cm_metrics.calcul_indc_matrice_confu(2)  # Classe 2 n'existe pas (il n'y a que 2 classes)

if __name__ == '__main__':
    unittest.main()
