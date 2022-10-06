import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier


def KNN(data, predict):

    # colonnes contient une liste des colonnes de mon fichier
    colonnes = []
    for i in range(len(data.columns)):
        colonnes.append(data.columns[i])
    # print(colonnes)

    # Changer les colonnes non-numériques en colonnes numériques
    le = preprocessing.LabelEncoder()

    # Traite les colonnes pour avoir que des chiffres
    for colonne in colonnes:
        data[colonne] = le.fit_transform(list(data[colonne]))


    # Consolider les données
    # retire la colonne à prédire de data et la met dans predict
    predict = data.pop(predict)
    #print(predict)

    # mise au bon format
    data = np.array(data)
    predict = np.array(predict)

    # Séparation des données en base d'entrainement / base de test
    data_train, data_test, predict_train, predict_test = model_selection.train_test_split(data, predict, test_size=0.1)

    # Model utilisé avec le nombre de voisins
    model = KNeighborsClassifier(n_neighbors=5)

    # Entrainement du modèle
    model.fit(data_train, predict_train)

    # Précision du modèle
    acc = model.score(data_test, predict_test)
    print("Accuracy : " + str(acc))

    # predicted reçoit les valeurs prédites par le modèle
    predicted = model.predict(data_test)

    # Permet de comparer les valeurs prédites avec les valeurs réelles
    for x in range(len(predicted)):
        print("Predicted: ", predicted[x], "Actual: ", predict_test[x])


def opti_KNN(data, predict, min_NN, max_NN):

    # colonnes contient une liste des colonnes de mon fichier
    colonnes = []
    for i in range(len(data.columns)):
        colonnes.append(data.columns[i])
    # print(colonnes)


    # Changer les colonnes non-numériques en colonnes numériques
    le = preprocessing.LabelEncoder()

    # Traite les colonnes pour avoir que des chiffres
    for colonne in colonnes:
        data[colonne] = le.fit_transform(list(data[colonne]))


    # Consolider les données
    # retire la colonne à prédire de data et la met dans predict
    predict = data.pop(predict)
    #print(predict)

    # mise au bon format
    data = np.array(data)
    predict = np.array(predict)

    # Valeur des acc pour des neighbors allant de min_NN à max_NN
    acc = np.zeros(max_NN - min_NN + 1)
    print(data)

    nbr_simu = 1000
    for simu in range(nbr_simu):
        for n_neigh in range(min_NN, max_NN + 1):
            # Séparation des données en base d'entrainement / base de test
            data_train, data_test, predict_train, predict_test = model_selection.train_test_split(data, predict,
                                                                                                  test_size=0.1)
            # Model utilisé avec le nombre de voisins
            model = KNeighborsClassifier(n_neighbors=n_neigh)

            # Entrainement du modèle
            model.fit(data_train, predict_train)

            # Précision du modèle
            acc[n_neigh - min_NN] += model.score(data_test, predict_test)

    acc = acc / nbr_simu
    print("Accuracy : " + str(acc))
    plt.plot([x for x in range(min_NN, max_NN + 1)], acc)
    plt.ylim(bottom=0)
    plt.show()


if __name__ == "__main__":
    print("blabla")