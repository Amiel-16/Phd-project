import torch

class ConfusionMatrixCalculator:
    def __init__(self, n_classes):
        """
        Initialise l'objet avec le nombre de classes pour la matrice de confusion.

        :param n_classes: Nombre de classes.
        """
        if n_classes <= 0:
            raise ValueError("Le nombre de classes doit être supérieur à 0.")

        self.n_classes = n_classes
        self.conf_matrix = torch.zeros(n_classes, n_classes, dtype=torch.int)

    def _validate_inputs(self, preds, labels):
        """
        Vérifie la validité des prédictions et des étiquettes.

        :param preds: Tensor contenant les prédictions (probabilités ou scores de modèle).
        :param labels: Tensor contenant les étiquettes réelles.
        """
        if not isinstance(preds, torch.Tensor):
            raise TypeError("Les prédictions doivent être un torch.Tensor.")
        if not isinstance(labels, torch.Tensor):
            raise TypeError("Les étiquettes doivent être un torch.Tensor.")

        if preds.size(0) != labels.size(0):
            raise ValueError("Le nombre d'exemples dans les prédictions et les étiquettes doit être le même.")

        if preds.size(1) != self.n_classes:
            raise ValueError(
                f"Les prédictions doivent avoir une taille ({preds.size(0)}, {self.n_classes}) correspondant au nombre de classes.")

    def update_confusion_matrix(self, preds, labels):
        """
        Met à jour la matrice de confusion à partir des prédictions et des étiquettes réelles.

        :param preds: Tensor contenant les prédictions (logits ou scores de modèle).
        :param labels: Tensor contenant les étiquettes réelles.

        :return: Matrice de confusion mise à jour.
        """
        # Validation des entrées
        self._validate_inputs(preds, labels)

        # Calcul des classes prédites
        preds = torch.argmax(preds, dim=1)

        # Mise à jour de la matrice de confusion
        for p, t in zip(preds, labels):
            self.conf_matrix[p.item(), t.item()] += 1

        return self.conf_matrix

