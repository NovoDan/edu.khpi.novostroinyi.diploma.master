import keras.layers as L
import keras.models as M
import neurokit2 as nk
import numpy as np
from keras.layers import Dropout
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from utils import convert_samples_to_milis


def hr_analysis(peaks, freq):
    """
    Розрахунок параметрів серцвого ритму
    Приймає індекси зубців та частоту дискретизації сигналу
    Повертає словник з розрахованими параметрами
    """
    i = 0
    rr_intervals = []
    # Розрахунок RR інтервалів
    while i < len(peaks) - 1:
        rr_intervals.append(peaks[i + 1] - peaks[i])
        i += 1
    # Перетворення значень інтервалів зі зразків у мілісекунди
    rr_intervals = convert_samples_to_milis(peaks, freq)
    hr = 60000 / rr_intervals  # значення серцевого ритму для кожного з інтервалів
    mean_hr = np.mean(hr)  # середнє значення серцевого ритму
    std_hr = np.std(hr)  # стандартне значення серцевого ритму
    return {'mHR': mean_hr, 'stdHR': std_hr}


def hrv_domain_analysis(peaks, fs):
    """
    Розрахунок доменних параметрів ВСР
    Приймає індекси зубців, чатоту дискретизації
    Повертає словник з параметрами ВСР
    """
    hrv_params = nk.hrv(peaks, fs)
    return hrv_params


def neuro_processing(x_train, y_train, x_val, y_val):
    """
    Модель нейронної мережі для аналізу ВСР
    Приймає масиви навчальних та валідаційних даних
    повертає навчені моделі
    """
    model = M.Sequential()
    model.add(Dropout(0.2))
    model.add(L.BatchNormalization())
    # Вхідний шар має 9 нейронів - по кількості вхідних параметрів
    model.add(L.Dense(units=9, activation='elu'))
    model.add(L.Dense(units=18, activation='sigmoid'))
    # Вихідний шар має 4 нейронів - по кількості класів для класифікації
    # softmax - дозволяє отриматі імовірність прилежності до класу
    model.add(L.Dense(units=4, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']  # відсоток правильних відповідей
    )

    # Навчання моделі
    print('Training model')
    history = model.fit(
        x_train,
        y_train,
        batch_size=32,  # 32 об'єкту для розрахунку градієнту на кожному кроці
        epochs=50,  # 50 проходів по датасету
        validation_data=(x_val, y_val)
    )

    knn = _classify(x_train, y_train, x_val, y_val)

    # друкуємо точність навчання
    plt.subplot(111)
    plt.title('Точність')
    plt.plot(history.history['accuracy'], label='Навчальні')
    plt.plot(history.history['val_accuracy'], label='Валідаційні')
    plt.legend()
    plt.show()

    model.save("/model/trained_model.hdf5")

    return model, knn


def _classify(x_train, y_train, x_val, y_val):
    neighbors = np.arange(1, 10)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    knn = None

    # Цикл з різними значеннями числа сусідів
    for i, k in enumerate(neighbors):
        # Ініціалізація об'єкту класифікатора з К сусідями
        knn = KNeighborsClassifier(n_neighbors=k)

        # Ззавантаження навчальних даних у класифікатор
        knn.fit(x_train, y_train)

        # Розрахунок точності для тестових даних
        train_accuracy[i] = knn.score(x_train, y_train)

        # Розрахунок точності для тренувальних даних
        test_accuracy[i] = knn.score(x_val, y_val)

    # Друкування графіку
    plt.title('Точність К-найближчих')
    plt.plot(neighbors, train_accuracy, label='Навчання')
    plt.plot(neighbors, test_accuracy, label='Тестування')
    plt.legend()
    plt.xlabel('Число сусідів, n')
    plt.ylabel('Точність, %')
    plt.show()
    return knn
