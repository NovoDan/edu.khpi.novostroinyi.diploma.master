import numpy as np
from wfdb import processing as wproc

from data_loader import read_ecg_from_csv
from hrv_analysis import hrv_domain_analysis, hr_analysis
from plotting import plot_ecg


def retrieve_test_ecg_data(path_to_ecg, delimiter, frequency, time_units, samp_from=0, samp_to=None, sig_units=['mV']):
    """
    Зчитування, обрізка та друк отриманого сигналу ЕКГ
    Повертає Обратий інтервал кардіограми
    """
    ecg_signal = read_ecg_from_csv(path_to_ecg, delimiter)
    # Якщо вказано - вирізати інтервал зі всієї довжини сигналу
    ecg_interval = ecg_signal[samp_from: samp_to, 1]
    # Друк ЕКГ
    plot_ecg(signal=ecg_interval,
             freq=frequency, title=f'ECG from {path_to_ecg}', sig_units=sig_units, time_units=time_units)

    return ecg_interval


def detect_peaks(ecg_signal, frequency):
    """
    Визначення R-зубців методом XQRS
    """
    xqrs = wproc.XQRS(ecg_signal, frequency)
    xqrs.detect()
    return xqrs.qrs_inds


def combine_hrv_params(peaks, freq):
    """
    Визначення параметрів серцевого ритму та ВСР
    Комбінація їх у єдиний масив
    Повертає масив параметрів ВСР
    """
    hr_params = hr_analysis(peaks, freq)
    hrv_features = hrv_domain_analysis(peaks, freq)
    hrv_test_features = [[hr_params['mHR'],
                          hr_params['stdHR'],
                          hrv_features['HRV_SDNN'].values[0],
                          hrv_features['HRV_MeanNN'].values[0],
                          hrv_features['HRV_MadNN'].values[0],
                          hrv_features['HRV_pNN50'].values[0],
                          hrv_features['HRV_RMSSD'].values[0],
                          hrv_features['HRV_ShanEn'].values[0],
                          hrv_features['HRV_SampEn'].values[0]]]
    return np.array(hrv_test_features)
