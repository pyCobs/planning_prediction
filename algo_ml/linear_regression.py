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