def calculate_rri(rr_indices, sampling_rate):
    """
    Розрахунок RR-інтервалів
    Приймає індекси R зубців та частоту дискретизації
    Повертає масив RR-інтервалів
    """
    rr_indices = convert_samples_to_milis(rr_indices, sampling_rate)
    i = 0
    rr_intervals = []
    while i < len(rr_indices) - 1:
        rr_intervals.append(rr_indices[i + 1] - rr_indices[i])
        i += 1
    return rr_intervals


def convert_samples_to_milis(items, freq=360):
    """
    Конвертація зразків у мілісекунди
    Приймає індекси зразців та частоту дискретизації
    Повертає конвертовані дані
    """
    for i in items:
        i *= 1000 / freq
    return items
