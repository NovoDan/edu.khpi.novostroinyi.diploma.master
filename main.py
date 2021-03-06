import preprocessing
from data_loader import load_train_data, create_datasets
from hrv_analysis import neuro_processing
from plotting import plot_ecg, plot_spectrum_density
from utils import calculate_rri, convert_samples_to_milis

model = None
knn = None


def test(directory, record, fs, samp_to):
    """
    Метод для здійснення аналізу ЕКГ
    Приймає шлях до файлу для аналізу, назву файлу, частоту дискретизації, довжину сигналу
    """
    global model, knn

    # Якщо нема навченої моделі - запустити навчання
    if model is None:
        model, knn = train_nn()

    # Зчитування тестового файлу
    path_to_file = directory + f'{record}.csv'
    signal = preprocessing.retrieve_test_ecg_data(path_to_file, ',', fs, time_units='seconds',
                                                  samp_to=samp_to)
    plot_ecg(record, fs)
    # Попередня обробка даних та друк результатів
    r_peaks = preprocessing.detect_peaks(ecg_signal=signal, frequency=fs)
    rr_intervals = calculate_rri(r_peaks, fs)
    rr_intervals_milis = convert_samples_to_milis(rr_intervals)
    plot_spectrum_density(rr_intervals_milis)
    test_hrv_params = preprocessing.combine_hrv_params(r_peaks, fs)

    print(f"Параметри для запису {record}:")
    print(test_hrv_params)
    print(f"Результати для запису {record}:")
    print(model.predict(test_hrv_params))  # аналіз нейронною мережею
    print(knn.predict(test_hrv_params))  # аналіз методом к-найближчих


def train_nn():
    """
    Навчання моделі
    Повертає навчену модель
    """
    # Зчитування даних для навчання
    train_data, train_labels = load_train_data()
    # Розділення даних на датасети
    x_train, y_train, x_val, y_val = create_datasets(train_data, train_labels, 0.2)
    return neuro_processing(x_train, y_train, x_val, y_val)


if __name__ == '__main__':
    # запуск програми
    sampling_frequency = 360
    samples_to = 100000

    directory_base = 'resources/'
    records = ['106', '113', '114', '115', '116', '117']
    test(directory_base, records, sampling_frequency, samples_to)
