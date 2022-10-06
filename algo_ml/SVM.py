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