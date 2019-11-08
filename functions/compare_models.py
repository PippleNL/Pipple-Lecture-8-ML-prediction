from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split

from functions.data_preparation import get_y_x, impute_x, labelencode_x, onehotencode_x, labelencode_y, feature_scaling


def make_clf(model_name, random_state=69, multi_class=False, feature_scaling_method='auto'):
    """
    Creates a classifier and determines which type of feature scaling should be applied if feature_scaling is set to
    'auto'.
    All based on the sklearn library.

    :param model_name: The classifier to create (string); either 'svc', 'RandomForest', 'knn', 'naive', 'logistic'
    :param random_state: The random state to apply (default 69); necessary for reproducibility
    :param multi_class: Boolean specifying whether it is a multi-class classification problem; only applicable when
    creating a logistic regression (default False)
    :param feature_scaling_method: Which type of feature scaling to apply; if set to 'auto', the default scaling per
    classifier is returned (default 'auto')
    :return: Classifier object and corresponding feature scaling method (string)
    """
    method = feature_scaling_method

    if model_name == 'svc':
        model = SVC(kernel='rbf',
                    random_state=random_state)
        if feature_scaling_method == 'auto':
            method = 'standardisation'
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100,
                                       random_state=random_state)
        if feature_scaling_method == 'auto':
            method = 'standardisation'
    elif model_name == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
        if feature_scaling_method == 'auto':
            method = 'standardisation'
    elif model_name == 'naive':
        model = GaussianNB()
        if feature_scaling_method == 'auto':
            method = 'standardisation'
    elif model_name == 'logistic':
        if multi_class:
            model = LogisticRegression(solver='lbfgs',
                                       multi_class='multinomial')
        else:
            model = LogisticRegression(solver='lbfgs',
                                       multi_class='ovr')
        if feature_scaling_method == 'auto':
            method = 'standardisation'
    else:
        raise ValueError('Unknown classification model: {}'.format(model_name))

    return model, method


def make_regr(model_name, feature_scaling_method='auto'):
    """
    Creates a regression and determines which type of feature scaling should be applied if feature_scaling is set to
    'auto'.
    All based on the sklearn library.

    :param model_name: The classifier to create (string); either 'svc', 'RandomForest', 'knn', 'naive', 'logistic'
    :param feature_scaling_method: Which type of feature scaling to apply; if set to 'auto', the default scaling per
    regression is returned (default 'auto')
    :return: Regressor object and corresponding feature scaling method (string)
    """
    method = feature_scaling_method

    from sklearn import linear_model, neighbors, tree, svm

    if model_name == 'OLS':
        model = linear_model.LinearRegression()
        if feature_scaling_method == 'auto':
            method = 'standardisation'
    elif model_name == 'Ridge':
        model = linear_model.Ridge()
        if feature_scaling_method == 'auto':
            method = 'standardisation'
    elif model_name == 'knn':
        model = neighbors.KNeighborsRegressor(5, weights='uniform')
        if feature_scaling_method == 'auto':
            method = 'standardisation'
    elif model_name == 'BayesianRidge':
        model = linear_model.BayesianRidge()
        if feature_scaling_method == 'auto':
            method = 'standardisation'
    elif model_name == 'DecisionTreeRegressor':
        model = tree.DecisionTreeRegressor(max_depth=1)
        if feature_scaling_method == 'auto':
            method = 'standardisation'
    elif model_name == 'SVM':
        model = svm.SVR()
        if feature_scaling_method == 'auto':
            method = 'standardisation'
    else:
        raise ValueError('Unknown classification model: {}'.format(model_name))

    return model, method


def sort_compute_metrics_clf(predictions, multi_class=False, sort_by='accuracy'):
    """
    Computes and sorts classification metrics.
    All based on the sklearn library.

    :param predictions: Iterable having a list that contains the classifier's name, np ndarray with predictions and an
    np ndarray with true labels
    :param multi_class: Boolean specifying whether it is a multi-class classification problem (default False)
    :param sort_by: The metric that is used as sorting criterion (string); either 'accuracy', 'precision', 'recall',
    'F1', 'auc' (default 'accuracy')
    :return: Header that lists which metrics re calculated for which model (tuple of strings) and a lists of metric
    scores (tuple of floats)
    """
    # retrieve header information
    header = ('model', 'accuracy', 'precision', 'recall', 'F1', 'auc')

    scores = []
    for model, y_pred, y_true in predictions:
        acc = accuracy_score(y_true, y_pred, normalize=True)

        if not multi_class:
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            scores.append((model, acc, precision[1], recall[1], f1[1], auc))
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            auc = '-'
            scores.append((model, acc, precision, recall, f1, auc))

    # sort metrics according to sort_by parameter
    scores = sorted(scores, key=lambda s: s[header.index(sort_by)], reverse=True)

    return header, scores


def sort_compute_metrics_regr(predictions, sort_by='r2'):
    """
    Computes and sorts regression metrics.
    All based on the sklearn library.

    :param predictions: Iterable having a list that contains the classifier's name, np ndarray with predictions and an
    np ndarray with true values
    :param sort_by: The metric that is used as sorting criterion (string);
    :return: Header that lists which metrics re calculated for which model (tuple of strings) and a lists of metric
    scores (tuple of floats)
    """
    # retrieve header information
    header = ('model', 'explained_variance_score', 'mse', 'mae', 'r2')
    from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error
    scores = []
    for model, y_pred, y_true in predictions:
        evs = explained_variance_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
        mse = mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
        mae = mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
        r2 = r2_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
        scores.append((model, evs, mse, mae, r2))

    # sort metrics according to sort_by parameter
    scores = sorted(scores, key=lambda s: s[header.index(sort_by)], reverse=True)

    return header, scores


def draw_confusion_matrix(predictions, classes, columns=1):
    """
    Draws a confusion matrix for all predictions invoked.
    All based on the sklearn library.

    :param predictions: Iterable having a list that contains the classifier's name, np ndarray with predictions and an
    np ndarray with true labels
    :param classes: Np ndarray that holds the label for each class in the dependent variable (i.e. y)
    :param columns: The number of plots (horizontally) drawn next to each other (default 2)
    :return: Nothing
    """
    plt.figure(figsize=(25, 15))
    rows = np.ceil(len(predictions) / columns)
    for i, (model, y_pred, y_true) in enumerate(predictions, 1):
        cm = confusion_matrix(y_true, y_pred)
        cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.subplot(rows, columns, i)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix - Model: {}'.format(model))
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for k, l in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(l, k, '{0:3d}\n{1:.2f}'.format(cm[k, l], cm2[k, l]),
                     horizontalalignment="center",
                     color="white" if cm[k, l] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    plt.show()


def draw_residual_plot(predictions):
    """
    Draws a residual plot for all predictions invoked.
    All based on the sklearn library.

    :param predictions: Iterable having a list that contains the model's name, np ndarray with predictions and an
    np ndarray with true values
    :return: Nothing
    """

    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt

    for i, (model, y_pred, y_true) in enumerate(predictions, 1):
        matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
        preds = pd.DataFrame({"preds": y_pred, "true": y_true})
        preds["residuals"] = preds["true"] - preds["preds"]
        preds.plot(x="preds", y="residuals", kind="scatter")
        plt.title('Residual plot - Model: {}'.format(model))

    plt.show()


def data_preprocessing(data, dependent, independent,
                       impute_strategy='drop', labelenc_x=None, onehotenc_x=None, labelenc_y=True):
    """
    Pre-processes the data such that the data gets prepared to be classified by some classifier. Pre-processing includes
    converting a pd dataframe into np ndarray, imputing missing values and label- or onehotencode dependent and
    independent variables.

    :param data: Pandas dataframe containing all problem information
    :param dependent: Column name tht will be used as dependent variable (i.e. y) in string
    :param independent: List of column names in string that will be used as independent variables (i.e. x)
    :param impute_strategy: The impute strategy; either a float or 'drop' (default), 'mean', 'median', 'most_frequent'
    :param labelenc_x: List of independent variables in string (e.g. column names) that need label-encoding; None if not
    applicable (default None)
    :param onehotenc_x: List of independent variables in string (e.g. column names) that need onehot-encoding; None if
    not applicable (default None)
    :param labelenc_y: Boolean specifying whether the dependent variable needs label-encoding (default True)
    :return: x - independent variables (np ndarray), y - dependent variable (np ndarray) and an np ndarray that holds
    the label for each class if labelenc_y is set True; None otherwise
    """
    # get y, x as np ndarray
    y, x, independent = get_y_x(data, dependent, independent)

    # impute x
    impute_features = data[independent].columns[data[independent].isnull().any()]
    impute_ind = [i for i, f in enumerate(independent) if f in impute_features]
    x = impute_x(x, impute_ind, impute_strategy)

    # label- or onehotencode x
    if labelenc_x is None:
        pass
    else:
        labelenc_ind = [i for i, f in enumerate(independent) if f in labelenc_x]
        x = labelencode_x(x, labelenc_ind)

    if onehotenc_x is None:
        pass
    elif len(onehotenc_x) == 0:
        pass
    else:
        onehotenc_ind = [i for i, f in enumerate(independent) if f in onehotenc_x]
        x = onehotencode_x(x, onehotenc_ind)

    # label-encode y
    classes = None
    if labelenc_y:
        y, classes = labelencode_y(y)

    return x, y, classes


def main_classification(data, dependent, independent,
                        impute_strategy='drop', labelenc_x=None, onehotenc_x=None, labelenc_y=True,
                        test_size=0.2, random_state=42,
                        models=('svc', 'RandomForest', 'knn', 'naive', 'logistic'),
                        feature_scaling_method='auto'):
    """
    Generates predictions for various classification models on a raw data set.
    All based on the sklearn library.

    :param data: Pandas dataframe containing all problem information
    :param dependent: Column name tht will be used as dependent variable (i.e. y) in string
    :param independent: List of column names in string that will be used as independent variables (i.e. x)
    :param impute_strategy: The impute strategy; either a float or 'drop' (default), 'mean', 'median', 'most_frequent'
    :param labelenc_x: List of independent variables in string (e.g. column names) that need label-encoding; None if not
    applicable (default None)
    :param onehotenc_x: List of independent variables in string (e.g. column names) that need onehot-encoding; None if
    not applicable (default None)
    :param labelenc_y: Boolean specifying whether the dependent variable needs label-encoding (default True)
    :param test_size: Float between 0 and 1 that specifies the proportion of the data set used to test the performance
    of a classifier (default 0.2)
    :param random_state: The random state to apply (default 42); necessary for reproducibility
    :param models: Iterable that lists the different classification models that will be compared; either 'svc',
    'RandomForest', 'knn', 'naive', 'logistic' (default ('svc', 'RandomForest', 'knn', 'naive', 'logistic'))
    :param feature_scaling_method: The feature scaling method to apply (string); either None, 'standardisation',
    'minmax' or 'auto' (default 'auto')
    :return: List having a list per classifier that contains the classifier's name, np ndarray with predictions and an
    np ndarray with true labels, an np ndarray that holds the label for each class if labelenc_y is set True; None
    otherwise
    """
    # pre-processing data
    x, y, classes = data_preprocessing(data, dependent, independent,
                                       impute_strategy=impute_strategy, labelenc_x=labelenc_x, onehotenc_x=onehotenc_x,
                                       labelenc_y=labelenc_y)

    # split x, y in train and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # determine type of classification
    multi_class = True if len(classes) > 2 else False

    # generate predictions
    predictions = []
    for model in models:
        # create classifier object and obtain desired feature scaling method
        classifier, fc_method = make_clf(model, random_state=random_state, multi_class=multi_class,
                                         feature_scaling_method=feature_scaling_method)

        # feature scale x accordingly
        x_train, x_test = feature_scaling(fc_method, x_train, x_test)

        # fit classifier
        classifier.fit(x_train, y_train)

        # generate predictions
        y_pred = classifier.predict(x_test)

        # store predictions
        predictions.append([model, y_pred, y_test])

    return predictions, classes


def main_regression(data, dependent, independent,
                    impute_strategy='drop', labelenc_x=None, onehotenc_x=None,
                    test_size=0.2, random_state=42,
                    models=('OLS', 'Ridge', 'knn', 'BayesianRidge', 'DecisionTreeRegressor', 'SVM'),
                    feature_scaling_method='auto'):
    """
    Generates predictions for various regression models on a raw data set.
    All based on the sklearn library.

    :param data: Pandas dataframe containing all problem information
    :param dependent: Column name tht will be used as dependent variable (i.e. y) in string
    :param independent: List of column names in string that will be used as independent variables (i.e. x)
    :param impute_strategy: The impute strategy; either a float or 'drop' (default), 'mean', 'median', 'most_frequent'
    :param labelenc_x: List of independent variables in string (e.g. column names) that need label-encoding; None if not
    applicable (default None)
    :param onehotenc_x: List of independent variables in string (e.g. column names) that need onehot-encoding; None if
    not applicable (default None)
    :param test_size: Float between 0 and 1 that specifies the proportion of the data set used to test the performance
    of a classifier (default 0.2)
    :param random_state: The random state to apply (default 42); necessary for reproducibility
    :param models: Iterable that lists the different classification models that will be compared; either 'OLS', 'Ridge',
    'knn', 'BayesianRidge', 'DecisionTreeRegressor', 'SVM' (default ('OLS', 'Ridge', 'knn', 'BayesianRidge',
    'DecisionTreeRegressor', 'SVM'))
    :param feature_scaling_method: The feature scaling method to apply (string); either None, 'standardisation',
    'minmax' or 'auto' (default 'auto')
    :return: List having a list per regression that contains the regressor's name, np ndarray with predictions and an
    np ndarray with true values
    """
    # pre-processing data
    x, y, classes = data_preprocessing(data, dependent, independent,
                                       impute_strategy=impute_strategy, labelenc_x=labelenc_x, onehotenc_x=onehotenc_x,
                                       labelenc_y=False)

    # split x, y in train and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # generate predictions
    predictions = []
    for model in models:
        # create classifier object and obtain desired feature scaling method
        regressor, fc_method = make_regr(model, feature_scaling_method=feature_scaling_method)

        # feature scale x accordingly
        x_train, x_test = feature_scaling(fc_method, x_train, x_test)

        # fit classifier
        regressor.fit(x_train, y_train)

        # generate predictions
        y_pred = regressor.predict(x_test)

        # round predictions
        y_pred = np.round(y_pred, 2)

        # store predictions
        predictions.append([model, y_pred, y_test])

    return predictions
