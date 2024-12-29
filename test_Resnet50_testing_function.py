"""

Le code présenté ici est la propriété de Amiel Allab et du Laboratoire de Recherche en Imagerie et Orthopédie LIO-ÉTS.
Il est exclusivement destiné à des fins de recrutement et ne doit en aucun cas être utilisé à d'autres fins sans autorisation préalable.

Toute utilisation, reproduction, ou diffusion du code en dehors de ce cadre est strictement interdite.
En accédant à ce code, vous acceptez de respecter ces conditions et de ne pas l'utiliser à des fins personnelles, commerciales
ou de recherche sans le consentement explicite du laboratoire LIO-ÉTS.

"""

from unittest import TestCase, main
from unittest.mock import patch
import os
from Resnet50_testing import check_directory_exists
from Resnet50_testing import check_model_exists

# Début de la classe de tests
class TestFunctions(TestCase):

    # Test de la fonction check_directory_exists
    @patch('os.path.exists')
    def test_check_directory_exists(self, mock_exists):
        # Test si le répertoire existe
        mock_exists.return_value = True
        result = check_directory_exists("test_dir")
        self.assertEqual(result, "test_dir")

        # Test si le répertoire n'existe pas
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            check_directory_exists("non_existent_dir")

    # Test de la fonction check_model_exists
    @patch('os.path.exists')
    def test_check_model_exists(self, mock_exists):
        # Test si le modèle existe
        mock_exists.return_value = True
        result = check_model_exists("model_path")
        self.assertEqual(result, "model_path")

        # Test si le modèle n'existe pas
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            check_model_exists("non_existent_model.pt")

# Lancer les tests
if __name__ == '__main__':
    main()

