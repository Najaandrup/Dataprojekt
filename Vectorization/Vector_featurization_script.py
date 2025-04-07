import numpy as np
import pandas as pd
import pod5 as p5
import matplotlib.pyplot as plt
import os
import sys

def moving_average(x, w):
    return np.convolve(x, np.ones(w)/w, mode='same')

def moving_variation(x, w):
    padded = np.pad(x, (w//2, w//2), mode='reflect')
    result = []
    for i in range(len(x)):
        window = padded[i:i+w]
        result.append(np.var(window))
    return np.array(result)

def remove_outliers(signal, threshold=3.0):
    mean = np.mean(signal)
    std = np.std(signal)
    return np.where(np.abs(signal - mean) > threshold * std, mean, signal)

def vectorize(start, end, signal, nwindows, window_size, outlier_threshold):
    first_quantile = int(start + (end - start) * 0.25)
    third_quantile = int(start + (end - start) * 0.75)
    signal50 = signal[first_quantile:third_quantile]
    signal50_mean = np.mean(signal50)
    normalized = signal / signal50_mean

    values = []
    for i in range(nwindows):
        x = i * ((end - start) / nwindows)
        x = int(x)
        start_index = max(int(start + x - (window_size // 2)), 0)
        end_index = min(int(start + x + (window_size // 2)), len(normalized))
        window = normalized[start_index:end_index]
        mean_value = np.mean(window)
        values.append(mean_value)

    vector = np.array(values)
    vector = remove_outliers(vector, threshold=outlier_threshold)

    mov_avg = moving_average(vector, 10)
    mov_var = moving_variation(vector, 20)

    feature_vector = np.stack([mov_avg, mov_var], axis=1)
    return feature_vector

def process_file(tsv_file, pod_file, nwindows, window_size, outlier_threshold, outdir):
    polyA_df = pd.read_csv(tsv_file, sep='\t', skiprows=1, skipfooter=3, engine='python')
    with p5.Reader(pod_file) as reader:
        pod_data = [(str(read.read_id), read.signal, read.pore) for read in reader.reads()]
    pod_df = pd.DataFrame(pod_data, columns=["read_id", "signal", "pore"])
    df = polyA_df.merge(pod_df, on="read_id")

    features = []
    for i in range(len(df)):
        start = df.iloc[i]['start']
        end = df.iloc[i]['end']
        signal = df.iloc[i]['signal']
        feature = vectorize(start, end, signal, nwindows, window_size, outlier_threshold)
        features.append(feature)

    feature_matrix = np.array(features)
    mean = np.mean(feature_matrix, axis=0)
    std = np.std(feature_matrix, axis=0)

    x = np.arange(mean.shape[0])  # x-akse til plots

    # Ekstraher et navn baseret på .pod5-filen
    base_name = os.path.basename(pod_file).split('.')[0]
    plot_name_avg = f"{base_name}_window_{nwindows}_moving_avg.png"
    plot_name_var = f"{base_name}_window_{nwindows}_moving_var.png"

    # Plot 1: Moving average
    plt.figure(figsize=(10, 5))
    plt.plot(x, mean[:, 0], label='Mean', color='blue')
    plt.fill_between(x, mean[:, 0]-std[:, 0], mean[:, 0]+std[:, 0], alpha=0.3, color='lightblue', label='±SD')
    plt.title('Moving Average ± SD')
    plt.xlabel('Window')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.savefig(os.path.join(outdir, plot_name_avg), dpi=300, bbox_inches='tight')

    # Plot 2: Moving variation
    plt.figure(figsize=(10, 5))
    plt.plot(x, mean[:, 1], label='Mean', color='green')
    plt.fill_between(x, mean[:, 1]-std[:, 1], mean[:, 1]+std[:, 1], alpha=0.3, color='lightgreen', label='±SD')
    plt.title('Moving Variation ± SD')
    plt.xlabel('Window')
    plt.ylabel('Variance')
    plt.legend()
    plt.savefig(os.path.join(outdir, plot_name_var), dpi=300, bbox_inches='tight')

    print(f"Plots gemt som:\n{plot_name_avg}\n{plot_name_var}")

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python vectorize_feature.py <tsv_file> <pod5_file> <nwindows> <window_size> <outlier_threshold> [<outdir>]")
        sys.exit(1)

    tsv_file = sys.argv[1]
    pod5_file = sys.argv[2]
    nwindows = int(sys.argv[3])
    window_size = int(sys.argv[4])
    outlier_threshold = float(sys.argv[5])
    outdir = sys.argv[6] if len(sys.argv) > 6 else "output"

    process_file(tsv_file, pod5_file, nwindows, window_size, outlier_threshold, outdir)