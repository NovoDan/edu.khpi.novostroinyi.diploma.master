import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def load_train_data():
    """
    Завантаження даних для навчання моделі
    Повертає: масив параметрів, масив міток
    """
    file = pd.read_csv('resources/data.csv')
    # Зчитуються перші 9 стовпців файлу - це стовпці значень
    features = file.take([0, 1, 2, 3, 4, 5, 6, 7, 8], axis=1)
    # Зчитуються мітки класів з останнього стовпця 'Class'
    classes = file['Class']

    return features.values, classes.values


def read_ecg_from_csv(path, delimiter):
    """
    Завантаження даних ЕКГ для аналізу
    Приймає шлях до файлу та символ-роздільник (',', ';', тощо)
    Повертає зчитані дані
    """
    return np.loadtxt(path, skiprows=1, delimiter=delimiter)


def create_datasets(values, labels, percent):
    """
    Формування наборів даних для навчання та перевірки моделі
    Приймає масиви парамерів, міток до них та відсоток тестової вибірки
    Повертає кортеж з розділених масивів
    """
    # Розділення даних на навчальний та тренувальний набори
    x_train, x_val, y_train, y_val = train_test_split(values, labels, test_size=percent)

    # Кодування міток класів у OneHot кодування
    le = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    # Перетворення символьних значень міток на числові
    y_train_int = le.fit_transform(y_train)
    y_val_int = le.fit_transform(y_val)
    # Зміна розмірності
    y_train_int_reshaped = y_train_int.reshape(len(y_train_int), 1)
    y_val_int_reshaped = y_val_int.reshape(len(y_val_int), 1)
    # Кодування
    y_train_oh = onehot_encoder.fit_transform(y_train_int_reshaped)
    y_val_oh = onehot_encoder.fit_transform(y_val_int_reshaped)

    return x_train, y_train_oh, x_val, y_val_oh
