import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler


def column2num(data, columns):
    """
    Converts a column's dtype to numeric.

    :param data: Pandas dataframe having the columns which need conversion
    :param columns: Iterable having the column names in string
    :return: Pandas dataframe including converted columns
    """
    for f in columns:
        data[f] = pd.to_numeric(data[f])

    return data


def get_y_x(data, dependent, independent):
    """
    Creates labels (i.e. y) and independent variables (i.e. x) from given column names as numpy ndarray.

    :param data: Pandas dataframe containing all problem information
    :param dependent: Column name tht will be used as dependent variable (i.e. y) in string
    :param independent: List of column names in string that will be used as independent variables (i.e. x)
    :return: y and x (both as np ndarray) and list of independent variables
    """
    if type(dependent) != str:
        raise ValueError(
            'Unsupported dependent variable: {}. \nPlease specify a column name in string that needs to be classified'
                .format(dependent))

    if type(independent) != list:
        raise ValueError(
            'Unsupported independent variable: {}. \nPlease specify a list of column names in string (or empty list) '
            'that should be used to classify the dependent variable'.format(independent))

    y = data[dependent].values
    if len(independent) == 0:
        independent = [f for f in data.columns if f != dependent]

    x = data[independent].values

    return y, x, independent


def impute_x(x, indices, strategy='drop'):
    """
    Imputes x (i.e. independent variables) according to the given strategy.
    All based on the sklearn library.

    :param x: Independent variables (np ndarray)
    :param indices: List of column indices that need imputation
    :param strategy: The impute strategy; either a float or 'drop' (default), 'mean', 'median', 'most_frequent'
    :return: x including dropped or imputed columns (np ndarray)
    """
    if len(indices) < 1:
        return x

    if type(strategy) == float:
        x[pd.isnull(x)] = strategy
    elif strategy == 'drop':
        x = pd.DataFrame(x).dropna(axis=0, how='any', inplace=False).values
    elif strategy in ['mean', 'median', 'most_frequent']:
        imputer = Imputer(missing_values='NaN', strategy=strategy, axis=0)
        imputer.fit(x[:, indices])
        x[:, indices] = imputer.transform(x[:, indices])
    else:
        raise ValueError(
            'Unsupported impute strategy: {}. \nPlease specify the impute strategy as a float or one of the following '
            'strings: {}'.format(strategy, ['drop', 'mean', 'median', 'most_frequent']))

    return x


def labelencode_x(x, indices):
    """
    Encodes independent variables with values between 0 and number of classes - 1.
    All based on the sklearn library.

    :param x: Independent variables (np ndarray)
    :param indices: List of column indices that need to be label encoded
    :return: Independent variables including label encoded variables (np ndarray)
    """
    labelencoder_x = LabelEncoder()

    for i in indices:
        x[:, i] = labelencoder_x.fit_transform(x[:, i])

    return x


def onehotencode_x(x, indices):
    """
    Encodes independent variables as a one-hot numeric array (i.e. dummies 0 or 1).
    All based on the sklearn library.

    :param x: Independent variables (np ndarray)
    :param indices: List of column indices that need to be one-hot encoded
    :return: Independent variables including one-hot encoded variables, where dummy trap columns are removed
    (np ndarray)
    """
    dummy_trap_columns = []

    for i in indices:
        if type(x[0, i]) == str:
            labelencoder_x = LabelEncoder()
            x[:, i] = labelencoder_x.fit_transform(x[:, i])
            dummy_trap_columns.append(len(labelencoder_x.classes_))

    onehotencoder = OneHotEncoder(categorical_features=indices)
    x = onehotencoder.fit_transform(x).toarray()

    # remove dummy trap columns
    x = x[:, [f for f in range(x.shape[1]) if f not in np.cumsum([0] + dummy_trap_columns[:-1])]]

    return x


def labelencode_y(y):
    """
    Encodes the dependent variable y with values between 0 and number of classes -1.
    All based on the sklearn library.

    :param y: Dependent variable (np ndarray)
    :return: Label encoded dependent variable y (np ndarray) and np ndarray that holds the label for each class
    """
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    return y, labelencoder_y.classes_


def feature_scaling(method, x_train, x_test):
    """
    Scales the independent variables according the the method invoked.
    All based on the sklearn library.

    :param method: The scaling method to apply (string); either None, 'standardisation' or 'minmax'
    :param x_train: Independent variables that will be used to train the feature scaler (np ndarray)
    :param x_test: Independent variables that will be transformed based on the trained feature scaler
    :return: Feature scaled independent variables to train and test a classifier
    """
    if method is None:
        return x_train, x_test
    elif method == 'standardisation':
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.transform(x_test)
    elif method == 'minmax':
        sc_x = MinMaxScaler()
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.transform(x_test)
    else:
        raise ValueError(
            'Unknown feature scaling method: {}.\nCurrently supported methods are "standardisation" or "minmax".'
                .format(method))

    return x_train, x_test


if __name__ == '__main__':
    pass
