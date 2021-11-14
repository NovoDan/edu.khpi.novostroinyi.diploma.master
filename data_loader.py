import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def load_train_data():
    file = pd.read_csv('resources/data.csv')
    # Зчитуються перші 9 стовпців файлу - це стовпці значень
    features = file.take([0, 1, 2, 3, 4, 5, 6, 7, 8], axis=1)
    # Зчитуються мітки класів з останньюого стовпця 'Class'
    classes = file['Class']

    return features.values, classes.values


def read_ecg_from_csv(path, delimiter):
    return np.loadtxt(path, skiprows=1, delimiter=delimiter)


def create_datasets(values, labels, percent):
    # Розділення даних на навчальний та тренувальний набори
    x_train, x_val, y_train, y_val = train_test_split(values, labels, test_size=percent)

    # Кодування міток класів у OneHot кодування
    le = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    y_train_int = le.fit_transform(y_train)
    y_val_int = le.fit_transform(y_val)

    y_train_int_reshaped = y_train_int.reshape(len(y_train_int), 1)
    y_val_int_reshaped = y_val_int.reshape(len(y_val_int), 1)

    y_train_oh = onehot_encoder.fit_transform(y_train_int_reshaped)
    y_val_oh = onehot_encoder.fit_transform(y_val_int_reshaped)

    return x_train, y_train_oh, x_val, y_val_oh
