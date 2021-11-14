import matplotlib.pyplot as plt
import numpy as np
import wfdb
import pyhrv.frequency_domain as fd
from matplotlib import style
from matplotlib.patches import Ellipse
from typing import List


def plot_ecg(signal, freq, peaks=None, sig_units=None, title=None, time_units=None, samp_to=5000):
    """
    Друк ЕКГ
    """
    if peaks is not None:
        signal, cutted_peaks = _cut_signal(signal, peaks, samp_to)
    else:
        signal = signal[:samp_to]
    wfdb.plot_items(signal=signal[:samp_to], ann_samp=peaks, fs=freq, title=title, figsize=(10, 4),
                    ecg_grids=None,
                    time_units=time_units, sig_units=sig_units)


def plot_rr_intervals_graph(rr_intervals, rec_name):
    """
    Друк графіку RR інтервалів
    :param rr_intervals:
    :param rec_name:
    :return:
    """
    x_2 = np.cumsum(rr_intervals)

    plt.figure(figsize=(10, 7))

    plt.subplot(111)
    plt.title("RR інтервали")
    plt.plot(x_2, rr_intervals, markeredgewidth=0, marker="o", markersize=8)
    plt.xlabel("Час (мс)")
    plt.title("RR-інтервали (мс)")

    plt.show()
    plt.savefig("C:/Users/Antrakal/PycharmProjects/MasterDiploma/plot/Graph_" + rec_name)



def plot_distrib(nn_intervals, bin_length: int = 8):
    """
    Метод друкує гістограму розподілу RR інтервалів
    """

    max_nn_i = max(nn_intervals)
    min_nn_i = min(nn_intervals)

    plt.figure(figsize=(12, 8))
    plt.title("Розподіл RR Інтервалів", fontsize=20)
    plt.xlabel("Час (мс)", fontsize=15)
    plt.ylabel("Число RR Інтервалів", fontsize=15)
    plt.hist(nn_intervals, bins=range(min_nn_i - 10, max_nn_i + 10, bin_length), rwidth=0.8)
    plt.show()


def plot_poincare(nn_intervals: List[float], plot_sd_features: bool = True):
    """
    Друк графіку Пуанкаре / Лоренца для RR інтервалів
    """
    ax1 = nn_intervals[:-1]
    ax2 = nn_intervals[1:]

    # Розрахунок параметрів довжини, ширини та центру
    dict_sd1_sd2 = _get_poincare_plot_features(nn_intervals)
    sd1 = dict_sd1_sd2["sd1"]
    sd2 = dict_sd1_sd2["sd2"]
    mean_nni = np.mean(nn_intervals)

    # Друк параметрів
    style.use("seaborn-darkgrid")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plt.title("Скаттерограмма", fontsize=20)
    plt.xlabel('NN_n (мс)', fontsize=15)
    plt.ylabel('NN_n+1 (мс)', fontsize=15)
    plt.xlim(min(nn_intervals) - 10, max(nn_intervals) + 10)
    plt.ylim(min(nn_intervals) - 10, max(nn_intervals) + 10)

    # Графік Пуанкаре
    ax.scatter(ax1, ax2, c='b', s=2)

    if plot_sd_features:
        # Налаштування друку еліпсу
        ells = Ellipse(xy=(mean_nni, mean_nni), width=2 * sd2 + 1,
                       height=2 * sd1 + 1, angle=45, linewidth=2,
                       fill=False)
        ax.add_patch(ells)

        ells = Ellipse(xy=(mean_nni, mean_nni), width=2 * sd2,
                       height=2 * sd1, angle=45)
        ells.set_alpha(0.05)
        ells.set_facecolor("blue")
        ax.add_patch(ells)

        # Налаштування друку стрілок
        sd1_arrow = ax.arrow(mean_nni, mean_nni, -sd1 * np.sqrt(2) / 2, sd1 * np.sqrt(2) / 2,
                             linewidth=3, ec='r', fc="r", label="Ширина")
        sd2_arrow = ax.arrow(mean_nni, mean_nni, sd2 * np.sqrt(2) / 2, sd2 * np.sqrt(2) / 2,
                             linewidth=3, ec='g', fc="g", label="Довжина")

        plt.legend(handles=[sd1_arrow, sd2_arrow], fontsize=12, loc="best")
    plt.show()


def plot_spectrum_density(rri):
    """
    Метод друкує щільність спектру
    """
    # Розрахунок щільності спектру
    result = fd.welch_psd(nni=rri)

    # Друк додаткової інформації
    print(result['fft_peak'])


def _cut_signal(signal, peaks, samp_to=5000):
    signal = signal[:samp_to]
    cutted_peaks = []
    for peak in peaks:
        if peak < len(signal):
            cutted_peaks.append(peak)
        else:
            break
    lst = list()
    return signal, lst.append(np.array(cutted_peaks))


def _get_poincare_plot_features(nn_intervals: List[float]) -> dict:
    """
    Метод повертає словник з трьома парметрами з нелінійного доменнтого аналізу ВСР
    """
    diff_nn_intervals = np.diff(nn_intervals)
    # вимірює ширину хмари пуанкаре
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    # вимірює довжину хмари пуанкаре
    sd2 = np.sqrt(2 * np.std(nn_intervals, ddof=1) ** 2 - 0.5 * np.std(diff_nn_intervals, ddof=1) ** 2)
    ratio_sd2_sd1 = sd2 / sd1

    poincare_plot_features = {
        'sd1': sd1,
        'sd2': sd2,
        'ratio_sd2_sd1': ratio_sd2_sd1
    }

    return poincare_plot_features
