"""

Le code présenté ici est la propriété de Amiel Allab et du Laboratoire de Recherche en Imagerie et Orthopédie LIO-ÉTS.
Il est exclusivement destiné à des fins de recrutement et ne doit en aucun cas être utilisé à d'autres fins sans autorisation préalable.

Toute utilisation, reproduction, ou diffusion du code en dehors de ce cadre est strictement interdite.
En accédant à ce code, vous acceptez de respecter ces conditions et de ne pas l'utiliser à des fins personnelles, commerciales
ou de recherche sans le consentement explicite du laboratoire LIO-ÉTS.

"""
import torch

class ConfusionMatrixMetrics:
    def __init__(self, conf_matrix, n_classes):
        """
        Initialise la classe avec la matrice de confusion et le nombre de classes.

        :param conf_matrix: Tensor de taille (n_classes, n_classes) représentant la matrice de confusion.
        :param n_classes: Nombre total de classes.
        """
        self.conf_matrix = conf_matrix
        self.n_classes = n_classes

    def _check_validity(self):
        """
        Vérifie la validité de la matrice de confusion et des paramètres.
        """
        if not isinstance(self.conf_matrix, torch.Tensor):
            raise TypeError("La matrice de confusion doit être un torch.Tensor.")

        if self.conf_matrix.dim() != 2 or self.conf_matrix.size(0) != self.conf_matrix.size(1):
            raise ValueError(f"La matrice de confusion doit être carrée ({self.n_classes}x{self.n_classes}).")

        if self.n_classes <= 0 or self.conf_matrix.size(0) != self.n_classes:
            raise ValueError("Le nombre de classes doit correspondre à la dimension de la matrice de confusion.")

    def calcul_indc_matrice_confu(self, c):
        """
        Calcule les indicateurs de performance à partir d'une matrice de confusion pour une classe donnée.

        :param c: Classe pour laquelle on calcule les indicateurs.

        :return: (TP, TN, FP, FN, sensibilité, spécificité) pour la classe `c`.
        """
        # Vérification de la validité des entrées
        self._check_validity()

        # Vérification si l'indice de classe est valide
        if c < 0 or c >= self.n_classes:
            raise IndexError(f"L'indice de classe {c} est hors des limites (0 à {self.n_classes - 1}).")

        try:
            TP = self.conf_matrix.diag()[c]  # Vrai positif pour la classe c
            idx = torch.arange(self.n_classes)
            idx = idx != c  # Index des autres classes pour TN, FP, FN

            # Calcul des indices TN, FP, FN
            TN = self.conf_matrix[idx, :][:, idx].sum()  # True Negatives
            FP = self.conf_matrix[c, idx].sum()  # False Positives
            FN = self.conf_matrix[idx, c].sum()  # False Negatives

            # Calcul des indicateurs
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensibilité (Recall)
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # Spécificité

            return TP, TN, FP, FN, sensitivity, specificity

        except Exception as e:
            print(f"Erreur lors du calcul des indicateurs pour la classe {c}: {e}")
            return 0, 0, 0, 0, 0, 0
