


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