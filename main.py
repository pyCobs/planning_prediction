import pandas as pd
from KNN import
#Import des données (Dataframe)
data = pd.read_csv("Data/Source/T0_VGR.csv", sep=";")

#Colonnes
'''
data = data[["Indicateur type d'affaire", "Phase du projet", "Nombre d'OTPb actifs", "Nombre d'OTPi actifs", "Nb d'OTPb + OTPi actifs",
             "Nb. Plannings maintenus L0", "Nb. Plannings maintenus L1", "Nb. Plannings maintenus L2", "Nb. Plannings maintenus L3", "Nb. Plannings maintenus",
             "Nb. ETP sur 2020", "Part achat (%)", "Part frais (%)", "Part heure (%)", "Part achats (k€)", "Part frais (k€)", "Part heures (k€)", "CATC (k€)",
             "Flux de dépenses mensuel (k€/mois)", "PMO", "Temps planning (%)", "Temps couts (%)", "Temps ressource (%)", "Temps planning (h/mois)",
             "Temps coût (h/mois)", "Temps Ressource (h/mois)", "Heures mensuelles", "Périmètre"]]
'''

# colonne que je veux prédire
predict = "Cat_T0 VGR"

# Modèles :
    
# K-nearest neighbors
# ATTENTION : KNN prend des données labelisées, pas de données continues
#KNN(data, predict)

# Support Vector Machine
# ATTENTION : SVM prédit des valeurs entières, pas de données continues
# SVM(data, predict)

# Linear Regression
#linear_regression(data, predict)

# Neural Network
# ATTENTION : il y a autant de neurones en sortie que de valeurs distinctes dans predict (max 13)
# ne pas utiliser pour prédire des valeurs continues
#NN(data, predict)
#opti_NN(data, predict)
#use_NN(data, "my_model_0.767", predict)

# Decision tree
# ATTENTION : toutes les données doivent être numériques
decision_Tree(data, predict)
#opti_decision_Tree(data, predict)