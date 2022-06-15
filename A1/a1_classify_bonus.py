#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return np.trace(C) / np.sum(C)


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    return [C[k, k] / np.sum(C[k, :]) for k in range(len(C))]


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    return [C[k, k] / np.sum(C[:, k]) for k in range(len(C))]
    

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    print('TODO Section 3.1')
    parameters = []
    penalty = ['l2', 'l1', 'elasticnet']
    l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_estimators = [5,10,50,100,200]
    # Number of features to consider at every split
    max_features = ['auto']
    # Maximum number of levels in tree
    max_depth = [3, 5,20,100,200]
    # Minimum number of samples required to split a node
    min_samples_split = [2,5,10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1,2,5]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    hidden_layer_sizes = [1,3,5,10,20]
    activation = ['logistic', 'tanh', 'relu']
    alpha = [0.25, 0.5, 0.75, 1]
    batch_size = [5, 15, 50, 100]
    learning_rate_init = [0.001, 0.0001, 0.01]
    learning_rate = ['adaptive', 'constant', 'invscaling']

    parameters.append({'loss': ['log', 'hinge', 'huber'],
                      'penalty': penalty,
                      "l1_ratio": l1_ratio,
                      'class_weight': [None, 'balanced']})
    parameters.append({'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-10, 1e-11]})
    parameters.append( {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap})
    parameters.append( {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'alpha': alpha,
        'batch_size': batch_size,
        'learning_rate_init': learning_rate_init,
        'learning_rate' : learning_rate
    })

    parameters.append({'n_estimators': n_estimators,
                     'learning_rate': [0.1, 0.001, 0.00001]})
    searched = [None] * 5
    best = [None] * 5
    for i, [name, classifier] in enumerate(classifiers):
        print("Training: " + name)
        class_n = classifier
        searched[i] = RandomizedSearchCV(class_n, parameters[i], cv=5, n_iter =15)
        searched[i].fit(X_train, y_train)
        best[i] = searched[i].best_estimator_
        best[i].fit(X_train, y_train)

    with open(f"{output_dir}/a1_bonus.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        for i, [name, classifier] in enumerate(classifiers):
            print("Predicting: " + name)
            y_pred = best[i].predict(X_test)
            confusion = confusion_matrix(y_test, y_pred)

            classifier_name = name
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            acc = accuracy(confusion)
            print("Accuracy")
            print(acc)
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            rec = recall(confusion)
            print("Recall")
            print(rec)
            outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
            prec = precision(confusion)
            print("Precision")
            print(prec)
            outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
            conf_matrix = confusion
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

    iBest = 4 # Adaboost
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print('TODO Section 3.2')

    X_1k = None
    y_1k = None
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        sample_sizes = [1000, 5000, 10000, 15000, 20000]
        name, classifier = classifiers[iBest]
        print("Using: " + name)
        for sample_size in sample_sizes:
            rand_indices = np.random.randint(0, X_train.shape[0], sample_size)
            X_train_sub = X_train[rand_indices, :]
            y_train_sub = y_train[rand_indices]
            print("Train with data size: " + str(sample_size))
            classifier.fit(X_train_sub, y_train_sub)
            y_pred = classifier.predict(X_test)
            confusion = confusion_matrix(y_test, y_pred)

            num_train = sample_size
            acc = accuracy(confusion)
            print("Accuracy")
            print(acc)
            outf.write(f'{num_train}: {acc:.4f}\n')

            if sample_size == 1000:
                X_1k = X_train_sub
                y_1k = y_train_sub


    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')
    name, classifier = classifiers[i]
    print("Using: " + name)
    
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        print("Training: all features, full dataset")
        classifier.fit(X_train, y_train)
        y_full_pred = classifier.predict(X_test)
        print("Training: all features, 1k data")
        classifier.fit(X_1k, y_1k)
        y_1k_pred = classifier.predict(X_test)

        print("Selecting: top 5 features")
        selector5 = SelectKBest(f_classif, k=5)
        print("Selecting: top 50 features")
        selector50 = SelectKBest(f_classif, k=50)
        X_train_5_full = selector5.fit_transform(X_train, y_train)
        X_test_5_full = selector5.transform(X_test)
        top5_full = selector5.get_support(indices=True)
        X_train_5_1k = selector5.fit_transform(X_1k, y_1k)
        X_test_5_1k = selector5.transform(X_test)
        top5_1k = selector5.get_support(indices=True)
        pp5 = selector5.pvalues_
        X_train_50_full = selector50.fit_transform(X_train, y_train)
        X_test_50_full = selector50.transform(X_test)
        top50_full = selector50.get_support(indices=True)
        X_train_50_1k = selector50.fit_transform(X_1k, y_1k)
        X_test_50_1k = selector50.transform(X_test)
        top50_1k = selector50.get_support(indices=True)
        pp50 = selector50.pvalues_

        print("Training: 5 features, full data")
        classifier.fit(X_train_5_full, y_train)
        y_pred_5_full = classifier.predict(X_test_5_full)
        confusion_5_full = confusion_matrix(y_test, y_pred_5_full)
        print("Training: 5 features, 1k data")
        classifier.fit(X_train_5_1k, y_1k)
        y_pred_5_1k = classifier.predict(X_test_5_1k)
        confusion_5_1k = confusion_matrix(y_test, y_pred_5_1k)
        print("Training: 50 features, full data")
        classifier.fit(X_train_50_full, y_train)
        y_pred_50_full = classifier.predict(X_test_50_full)
        confusion_50_full = confusion_matrix(y_test, y_pred_50_full)
        print("Training: 5 features, 1k data")
        classifier.fit(X_train_50_1k, y_1k)
        y_pred_50_1k = classifier.predict(X_test_50_1k)
        confusion_50_1k = confusion_matrix(y_test, y_pred_50_1k)

        # for each number of features k_feat, write the p-values for
        # that number of features:
        k_feat = 5
        p_values = [v for i, v in enumerate(pp5) if i in top5_full]
        outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')
        k_feat = 50
        p_values = [v for i, v in enumerate(pp50) if i in top50_full]
        outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')

        accuracy_1k = accuracy(confusion_5_1k)
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        accuracy_full = accuracy(confusion_5_full)
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        feature_intersection = list(set(top5_full) &
                                    set(top5_1k))
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {top5_full}\n')
        pass


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        num_splits = 5
        kf = KFold(n_splits=num_splits, shuffle=True)
        X_merged = np.concatenate((X_train, X_test))
        y_merged = np.concatenate((y_train, y_test))
        accuracies = []
        # for each fold:
        for train_index, test_index in kf.split(X_merged, y_merged):
            kfold_accuracies = []
            for name, classifier in classifiers:
                print("[KFold] Training: " + name)
                classifier.fit(X_merged[train_index], y_merged[train_index])
                y_pred = classifier.predict(X_merged[test_index])
                confusion = confusion_matrix(y_merged[test_index], y_pred)
                kfold_accuracies.append(accuracy(confusion))
            print("[KFold] Accuracies: " + str(kfold_accuracies))
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
            # accuracies.append(0 if not kfold_accuracies else sum(kfold_accuracies) / len(kfold_accuracies))
            accuracies.append(kfold_accuracies)
        best_index = i
        best_name = classifiers[best_index][0]
        p_values = []
        np_accuracy = np.array(accuracies)
        for index, [name, classifier] in enumerate(classifiers):
            if index == best_index:
                continue
            s = ttest_rel(np_accuracy[:, index], np_accuracy[:, best_index])
            print(name + " vs " + best_name + ": p-value=" + str(s.pvalue))
            p_values.append(s.pvalue)
        outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')
            # pass


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    # TODO: load data and split into train and test.
    data = np.load(args.input)['arr_0']
    train, test = train_test_split(data, train_size=0.8, test_size=0.2)
    X_train = train[:, 0:173]
    y_train = train[:, 173]
    X_test = test[:, 0:173]
    y_test = test[:, 173]

    global classifiers
    classifiers = [
        ["sgd_classifier", SGDClassifier()],
        ["gaussian_nb", GaussianNB()],
        ["random_forest", RandomForestClassifier(n_estimators=10, max_depth=5)],
        ["mlp_classifier", MLPClassifier(alpha=0.05)],
        ["ada_boost", AdaBoostClassifier()]
    ]

    # TODO : complete each classification experiment, in sequence.
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    # X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    # class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    # class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)
