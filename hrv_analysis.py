import keras.layers as L
import keras.models as M
import neurokit2 as nk
import numpy as np
from keras.layers import Dropout
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from utils import convert_samples_to_milis


def hr_analysis(peaks, freq):
    i = 0
    rr_intervals = []
    while i < len(peaks) - 1:
        rr_intervals.append(peaks[i + 1] - peaks[i])
        i += 1

    rr_intervals = convert_samples_to_milis(peaks, freq)
    hr = 60000 / rr_intervals
    mean_hr = np.mean(hr)
    std_hr = np.std(hr)
    return {'mHR': mean_hr, 'stdHR': std_hr}


def hrv_domain_analysis(peaks, fs):
    hrv_params = nk.hrv(peaks, fs)
    return hrv_params


def neuro_processing(x_train, y_train, x_val, y_val):
    model = M.Sequential()
    model.add(Dropout(0.2))
    # Вхідний та прихований шари мають по 9 нейронів - по кількості вхідних параметрів
    model.add(L.BatchNormalization())
    model.add(L.Dense(units=9, activation='elu'))
    model.add(L.Dense(units=18, activation='sigmoid'))
    # Вихідний шар має 4 нейронів - по кількості класів віхідних даних
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

    knn = KNeighborsClassifier(4)
    _classify(knn, x_train, y_train)

    # друкуємо точність навчання
    plt.subplot(111)
    plt.title('Точність')
    plt.plot(history.history['accuracy'], label='Навчальні')
    plt.plot(history.history['val_accuracy'], label='Валідаційні')
    plt.legend()
    plt.show()

    model.save("C:/Users/Antrakal/PycharmProjects/MasterDiploma/model/trained_model.hdf5")

    return model, knn


def _classify(model, features, classes):
    kf = KFold(10, shuffle=True)
    score = []
    for trainIndex, testIndex in kf.split(features):
        xTrain, xTest = features[trainIndex], features[testIndex]
        yTrain, yTest = classes[trainIndex], classes[testIndex]

        model.fit(xTrain, yTrain)
        prediction = model.predict(xTest)
        score.append(accuracy_score(yTest, prediction))

    print("Середня точнсть методу К найближчих сусідів ", sum(score) / len(score))
