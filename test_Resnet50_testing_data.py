"""

Le code présenté ici est la propriété de Amiel Allab et du Laboratoire de Recherche en Imagerie et Orthopédie LIO-ÉTS.
Il est exclusivement destiné à des fins de recrutement et ne doit en aucun cas être utilisé à d'autres fins sans autorisation préalable.

Toute utilisation, reproduction, ou diffusion du code en dehors de ce cadre est strictement interdite.
En accédant à ce code, vous acceptez de respecter ces conditions et de ne pas l'utiliser à des fins personnelles, commerciales
ou de recherche sans le consentement explicite du laboratoire LIO-ÉTS.

"""

from unittest import TestCase, main
from torchvision import datasets, transforms
from unittest.mock import patch

class TestModelLoading(TestCase):

    # Test du chargement du jeu de données
    @patch('torchvision.datasets.ImageFolder')
    def test_data_loading(self, mock_imagefolder):
        mock_dataset = mock_imagefolder.return_value
        mock_dataset.__len__.return_value = 100
        test_dir = "C:/Dossier_travail/DATA/IMAGES/VAL_final_1/classif_GT/Training/Test"

        # Test si la longueur du dataset est correcte
        dataset = datasets.ImageFolder(test_dir)
        self.assertEqual(len(dataset), 100)

# Lancer les tests
if __name__ == '__main__':
    main()