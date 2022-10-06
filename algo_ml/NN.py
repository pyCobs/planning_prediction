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