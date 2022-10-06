import random
import pickle

import numpy as np
from numpy import unique
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import linear_model, preprocessing, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras

import pydotplus
from graphviz import Digraph
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg


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
    data_train, data_test, predict_train, predict_test = sklearn.model_selection.train_test_split(data, predict, test_size=0.1)

    # Model utilisé avec le nombre de voisins
    model = KNeighborsClassifier(n_neighbors=9)

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

def decision_Tree(data, predict_name):
    # colonnes contient une liste des colonnes de mon fichier
    colonnes = []
    for i in range(len(data.columns)):
        colonnes.append(data.columns[i])

    # Changer les colonnes non-numériques en colonnes numériques
    le = preprocessing.LabelEncoder()
    #print(data[predict_name])

    # Traite les colonnes pour avoir que des chiffres
    for colonne in colonnes:
        if colonne != predict_name:
            if isinstance(data[colonne][0], str):
                data[colonne] = le.fit_transform(list(data[colonne]))
    # data['Etat'] = le.fit_transform(list(data['Etat']))
    data[predict_name] = le.fit_transform(list(data[predict_name]))

    # Consolider les données
    # Retire la colonne à prédire de data et la met dans predict
    predict = data.pop(predict_name)
    #print(predict)

    # Mise au bon format

    predict_max = max(predict) + 1  # +1 si la série commence à zéro
    data = np.array(data)
    predict = np.array(predict)

    data_train, data_test, predict_train, predict_test = sklearn.model_selection.train_test_split(data, predict,
                                                                                                  test_size=0.1)
    colonnes.pop(colonnes.index(predict_name))

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(data_train, predict_train)
    data_tree = tree.export_graphviz(dtree, out_file=None, feature_names=colonnes)
    graph = pydotplus.graph_from_dot_data(data_tree)
    graph.write_png('mydecisiontree' + str(predict_name) + '.png')

    img = pltimg.imread('mydecisiontree' + str(predict_name) + '.png')
    imgplot = plt.imshow(img)
    #plt.show()


    #test_loss, test_acc = dtree.evaluate(data_test, predict_test)

    #print("Tested acc :", test_acc)
    # print(data_test)
    # print(predict_test)
    # predicted reçoit les valeurs prédites par le modèle
    # il y a une valeur pour chaque neurone en sortie avec la somme = 1

    predicted = dtree.predict(data_test)

    print("predict_test")
    print(predict_test)
    #print(le.inverse_transform(predict_test))

    print("predicted")
    print(predicted)
    #print(le.inverse_transform(predicted))


    # je calcule la précision de mon arbre
    precision = 0
    for pos in range(len(predicted)):
        if predicted[pos] == predict[pos]:
            precision += 1

    accuracy = precision / len(predicted)

    print("Acc : " + str(accuracy))

    print("Voulez-vous sauvegarder le modèle ?")
    entree = input("y/n : ")
    while not ((entree == "y") or (entree == "n")):
        print("Voulez-vous sauvegarder le modèle ?")
        entree = input("y/n : ")
    if entree == "y":
        with open("dtree_model_"+ str(predict_name) +".pkl", "wb") as f:
            pickle.dump(dtree, f)


def opti_decision_Tree(data, predict_name):
    # colonnes contient une liste des colonnes de mon fichier
    colonnes = []
    for i in range(len(data.columns)):
        colonnes.append(data.columns[i])

    # Changer les colonnes non-numériques en colonnes numériques
    le = preprocessing.LabelEncoder()
    #print(data[predict_name])

    # Traite les colonnes pour avoir que des chiffres
    for colonne in colonnes:
        if colonne != predict_name:
            if isinstance(data[colonne][0], str):
                data[colonne] = le.fit_transform(list(data[colonne]))
    # data['Etat'] = le.fit_transform(list(data['Etat']))
    data[predict_name] = le.fit_transform(list(data[predict_name]))

    # Consolider les données
    # Retire la colonne à prédire de data et la met dans predict
    predict = data.pop(predict_name)
    #print(predict)

    # Mise au bon format

    predict_max = max(predict) + 1  # +1 si la série commence à zéro
    data = np.array(data)
    predict = np.array(predict)
    precision_max = 0
    correlation_max = 0
    colonnes.pop(colonnes.index(predict_name))
    for _ in range(20):
        data_train, data_test, predict_train, predict_test = sklearn.model_selection.train_test_split(data, predict,
                                                                                                      test_size=0.1)
        dtree = DecisionTreeClassifier()
        dtree = dtree.fit(data_train, predict_train)

        predicted = dtree.predict(data_test)

        print("predict_test")
        print(predict_test)
        #print(le.inverse_transform(predict_test))

        print("predicted")
        print(predicted)
        #print(le.inverse_transform(predicted))


        # je calcule la précision de mon arbre
        precision = 0
        for pos in range(len(predicted)):
            if predicted[pos] == predict[pos]:
                precision += 1

        accuracy = precision / len(predicted)
        print("Acc : " + str(accuracy))

        # je calcule le coefficient de corrélation
        corr = np.corrcoef(predict_test, predicted)[0,1]
        print("Corrélation : " + str(corr))

        if accuracy > precision_max or corr > correlation_max:
            if accuracy > precision_max:
                precision_max = accuracy
            if corr > correlation_max:
                correlation_max = corr

            data_tree = tree.export_graphviz(dtree, out_file=None, feature_names=colonnes)
            graph = pydotplus.graph_from_dot_data(data_tree)
            graph.write_png('mydecisiontree' + str(predict_name) + '.png')

            img = pltimg.imread('mydecisiontree' + str(predict_name) + '.png')
            imgplot = plt.imshow(img)

            with open("dtree_model_"+ str(predict_name) + "_" + str(accuracy) + "_" + str(corr) +".pkl", "wb") as f:
                pickle.dump(dtree, f)


def SVM(data, predict):

    # colonnes contient une liste des colonnes de mon fichier
    colonnes = []
    for i in range(len(data.columns)):
        colonnes.append(data.columns[i])

    # Changer les colonnes non-numériques en colonnes numériques
    le = preprocessing.LabelEncoder()

    # Traite les colonnes pour avoir que des chiffres
    for colonne in colonnes:
        if isinstance(data[colonne][0], str):
            data[colonne] = le.fit_transform(list(data[colonne]))

    # Consolider les données
    # Retire la colonne à prédire de data et la met dans predict
    predict = data.pop(predict)

    # Mise au bon format
    data = np.array(data)
    predict = np.array(predict)

    data_train, data_test, predict_train, predict_test = sklearn.model_selection.train_test_split(data, predict,
                                                                                                  test_size=0.1)
    # Model utilisé
    clf = svm.SVC(kernel="linear")

    # Entrainement du model
    clf.fit(data_train, predict_train)

    # predicted contient les valeurs prédites par mon model pour data_test
    predicted = clf.predict(data_test)

    # Comparaison des valeurs prédites avec les valeurs réelles
    acc = metrics.accuracy_score(predict_test, predicted)
    print("predict_test :")
    print(predict_test)
    print("predicted :")
    print(predicted)
    print(acc)


def linear_regression(data, predict):

    # colonnes contient une liste des colonnes de mon fichier
    colonnes = []
    for i in range(len(data.columns)):
        colonnes.append(data.columns[i])

    # Changer les colonnes non-numériques en colonnes numériques
    le = preprocessing.LabelEncoder()

    # Traite les colonnes pour avoir que des chiffres
    for colonne in colonnes:
        if isinstance(data[colonne][0], str):
            data[colonne] = le.fit_transform(list(data[colonne]))

    # Consolider les données
    # Retire la colonne à prédire de data et la met dans predict
    predict = data.pop(predict)

    # Mise au bon format
    data = np.array(data)
    predict = np.array(predict)

    data_train, data_test, predict_train, predict_test = sklearn.model_selection.train_test_split(data, predict,
                                                                                                  test_size=0.1)
    linear = linear_model.LinearRegression()

    linear.fit(data_train, predict_train)
    acc = linear.score(data_test, predict_test)
    print("Acc : " + str(acc))

    # Coefficients de ma régression linéaire
    #print('Coefficient: \n', linear.coef_)
    #print('Intercept: \n', linear.intercept_)

    # predicted reçoit les valeurs prédites par le modèle
    predicted = linear.predict(data_test)

    # Comparaison des valeurs prédites avec les valeurs réelles
    for x in range(len(predicted)):
        print(predicted[x], predict_test[x])

def opti_NN(data, predict):
    # colonnes contient une liste des colonnes de mon fichier
    colonnes = []
    for i in range(len(data.columns)):
        colonnes.append(data.columns[i])
    nom = predict
    # Changer les colonnes non-numériques en colonnes numériques
    le = preprocessing.LabelEncoder()

    # Traite les colonnes pour avoir que des chiffres
    for colonne in colonnes:
        if colonne != predict:
            if isinstance(data[colonne][0], str):
                data[colonne] = le.fit_transform(list(data[colonne]))
    #data['Etat'] = le.fit_transform(list(data['Etat']))
    data[predict] = le.fit_transform(list(data[predict]))

    # Consolider les données
    # Retire la colonne à prédire de data et la met dans predict
    predict = data.pop(predict)

    # Mise au bon format

    #predict_unique = predict.unique() # si les valeurs à prédire n'ont pas de "trous"
    predict_max = max(predict) + 1  # +1 si la série commence à zéro
    data = np.array(data)
    predict = np.array(predict)
    test_acc_best = 0
    for _ in range(20):
        data_train, data_test, predict_train, predict_test = sklearn.model_selection.train_test_split(data, predict,
                                                                                                      test_size=0.1)
        a = random.randint(10, 2000)
        b = random.randint(10, 2000)
        model = keras.Sequential()
        model.add(keras.layers.Dense(len(data)))
        model.add(keras.layers.Dense(a, activation="relu"))
        model.add(keras.layers.Dense(b, activation="relu"))
        model.add(keras.layers.Dense(predict_max, activation="softmax"))

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        model.fit(data_train, predict_train, epochs=30)

        test_loss, test_acc = model.evaluate(data_test, predict_test)

        predicted = model.predict(data_test)
        prediction = []

        # je récupère la position de mon neurone avec la plus forte valeur
        for predic in predicted:
            prediction.append(list(predic).index(max(predic)))

        distinct = len(unique(prediction))

        if test_acc > test_acc_best and distinct >= 3:
            test_acc_best = test_acc
            model.save("my_model_" + str(nom) + "_" + str(test_acc))


def use_NN(data, model_path, predict):

    # colonnes contient une liste des colonnes de mon fichier
    colonnes = []
    for i in range(len(data.columns)):
        colonnes.append(data.columns[i])

    # Changer les colonnes non-numériques en colonnes numériques
    le = preprocessing.LabelEncoder()

    # Traite les colonnes pour avoir que des chiffres
    for colonne in colonnes:
        if colonne != predict:
            if isinstance(data[colonne][0], str):
                data[colonne] = le.fit_transform(list(data[colonne]))
    #data['Etat'] = le.fit_transform(list(data['Etat']))
    data[predict] = le.fit_transform(list(data[predict]))

    # Consolider les données
    # Retire la colonne à prédire de data et la met dans predict
    etat = data.pop(predict)
    etat = np.array(etat)

    model = keras.models.load_model(model_path)
    predicted = model.predict(data)
    prediction = []
    for predict in predicted:
        prediction.append(np.argmax(predict))

    prediction = np.array(prediction)

    compteur = 0
    for i in range(len(prediction)):
        if etat[i] == prediction[i]:
            compteur += 1
        print("Valeur : " + str(etat[i]) + "; Prediction : " + str(prediction[i]))

    print("Acc = " + str(compteur / len(prediction)))


def NN(data, predict):
    # colonnes contient une liste des colonnes de mon fichier
    colonnes = []
    for i in range(len(data.columns)):
        colonnes.append(data.columns[i])

    # Changer les colonnes non-numériques en colonnes numériques
    le = preprocessing.LabelEncoder()
    print(data[predict])
    # Traite les colonnes pour avoir que des chiffres
    for colonne in colonnes:
        if colonne != predict:
            if isinstance(data[colonne][0], str):
                data[colonne] = le.fit_transform(list(data[colonne]))
    #data['Etat'] = le.fit_transform(list(data['Etat']))
    data[predict] = le.fit_transform(list(data[predict]))

    # Consolider les données
    # Retire la colonne à prédire de data et la met dans predict
    predict = data.pop(predict)
    print(predict)

    # Mise au bon format

    predict_max = max(predict) + 1  # +1 si la série commence à zéro
    data = np.array(data)
    predict = np.array(predict)

    data_train, data_test, predict_train, predict_test = sklearn.model_selection.train_test_split(data, predict,
                                                                                                  test_size=0.1)

    model = keras.Sequential()
    model.add(keras.layers.Dense(len(data)))
    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dense(predict_max, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(data_train, predict_train, epochs=10)

    test_loss, test_acc = model.evaluate(data_test, predict_test)

    print("Tested acc :", test_acc)
    #print(data_test)
    #print(predict_test)
    # predicted reçoit les valeurs prédites par le modèle
    # il y a une valeur pour chaque neurone en sortie avec la somme = 1

    predicted = model.predict(data_test)
    prediction = []

    # je récupère la position de mon neurone avec la plus forte valeur
    for predic in predicted:
        prediction.append(list(predic).index(max(predic)))

    acc2 = np.corrcoef(predict_test, prediction)
    print("Acc linear : " + str(acc2[0,1]))

    print("predict_test")
    print(predict_test)
    print(le.inverse_transform(predict_test))
    print("prediction")
    print(prediction)
    print(le.inverse_transform(prediction))

    print("Voulez-vous sauvegarder le modèle ?")
    entree = input("y/n : ")
    while not((entree == "y") or (entree == "n")):
        print("Voulez-vous sauvegarder le modèle ?")
        entree = input("y/n : ")
    if entree == "y":
        model.save("my_model")
