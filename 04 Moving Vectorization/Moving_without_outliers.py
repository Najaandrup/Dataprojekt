import os
import sys
import numpy as np
import pandas as pd
import pod5 as p5
import matplotlib.pyplot as plt

def remove_outliers(signal, threshold):
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return signal
    z_scores = np.abs((signal - mean) / std)
    return signal[z_scores < threshold]  # returns signal if z-score is below the threshold

def vectorize_and_plot(tsv_file, pod_file, vector_length, window_size_avg, window_size_var, threshold):
    # Load position data
    polyA_df = pd.read_csv(tsv_file, sep='\t', skiprows=1, skipfooter=3, engine='python')
    polyA_df.columns = polyA_df.columns.str.strip()  # Fix possible whitespace in headers

    # Load signal data from pod5 file
    with p5.Reader(pod_file) as reader:
        pod_data = [(str(read.read_id), read.signal) for read in reader.reads()]
    pod_df = pd.DataFrame(pod_data, columns=["read_id", "signal"])

    # Merge dataframes
    df = polyA_df.merge(pod_df, on="read_id")

    # Number of reads
    n_rows = len(df)

    # Empty lists for all means and variances of the dataset
    all_means = []
    all_variances = []

    # Loop over each row / read
    for _, row in df.iterrows():
        raw_signal = row['signal']
        start = row['start']
        end = row['end']

        # Only takes signal of PolyA-tail
        signal = raw_signal[start:end]

        # Uses function above to remove outlies from PolyA-tail signal
        signal = remove_outliers(signal, threshold)

        # Compute 25-75% quantile region
        first_quantile = int(len(signal) * 0.25)
        third_quantile = int(len(signal) * 0.75)

        # Finds the signal in the 50% region
        signal50 = signal[first_quantile:third_quantile]

        # Checks that signal50 is not empty
        if len(signal50) == 0:
            continue  

        # Computes the mean of signal50
        signal50_mean = np.mean(signal50)

        # Checks that mean is not 0 before dividing
        if signal50_mean == 0:
            continue  

        # Divides signal by the mean of signal50
        normalized = signal / signal50_mean

        # Empty lists for the means and variances
        means = []
        variances = []

        # Loop over lenght of the vector 
        for i in range(vector_length):
            # Calculates evenly spaced position 'x' along the signal
            x = int(i * (len(signal) / vector_length))

            # Defines the start and end index of the window around position x
            start_index_avg = max(int(x - (window_size_avg // 2)), 0)  # start is 0 if x - half of the window size is negative
            end_index_avg = min(int(x + (window_size_avg // 2)), len(normalized))  # end is length of normalized signal if x + half of the window size is larger than the length

            # Defines the window around x
            window_avg = normalized[start_index_avg:end_index_avg]  

            # Appends the correct mean value of the window to the list 'means'
            if len(window_avg) == 0:
                means.append(np.nan)
            else:
                means.append(np.mean(window_avg))
            
            # Repeats for the var values
            start_index_var = max(int(x - (window_size_var // 2)), 0)
            end_index_var = min(int(x + (window_size_var // 2)), len(normalized))

            window_var = normalized[start_index_var:end_index_var]

            if len(window_var) == 0:
                variances.append(np.nan)
            else:
                variances.append(np.var(window_var))

        # Appends values from lists 'means' and 'variances' to lists with all the values
        all_means.append(means)
        all_variances.append(variances)

    # Convert to arrays
    all_means_array = np.array(all_means)
    all_variances_array = np.array(all_variances)

    # Calculates the mean and standard deviation for each mean feature position while ignoring NaNs
    avg_means = np.nanmean(all_means_array, axis=0)
    std_means = np.nanstd(all_means_array, axis=0)

    # Calculates the mean and standard deviation for each variance feature position while ignoring NaNs
    avg_vars = np.nanmean(all_variances_array, axis=0)
    std_vars = np.nanstd(all_variances_array, axis=0)

    # Creates an array for the x-axis
    x = np.arange(all_means_array.shape[1])  # window indices

    # Get base filename
    base_name = os.path.splitext(os.path.basename(tsv_file))[0]
    out_dir = './output/Moving_without_outliers_2_pipeline_output/'

    # Plot 1: Mean
    plt.figure(figsize=(10, 5))
    plt.plot(x, avg_means, label='Average Mean', color='blue')
    plt.fill_between(x, avg_means - std_means, avg_means + std_means, 
                     color='blue', alpha=0.2, label='±1 SD')
    plt.title(f"Moving Average {base_name} - Normalized\n" 
              f"vector length{vector_length}, window size{window_size_avg}, max Z score {threshold}, Number of datasets {n_rows}")
    plt.xlabel("Window Index")
    plt.ylabel("Normalized Mean")
    plt.legend()
    plt.grid(True)
    mean_plot_name = f"{base_name}_mean_n{vector_length}_w_mean{window_size_avg}_w_var{window_size_var}_t{threshold}_outlier_removed.png"
    plt.savefig(out_dir + mean_plot_name, dpi=300)
    plt.close()

    # Plot 2: Variance
    plt.figure(figsize=(10, 5))
    plt.plot(x, avg_vars, label='Average variance', color='orange')
    plt.fill_between(x, avg_vars - std_vars, avg_vars + std_vars, 
                     color='orange', alpha=0.2, label='±1 SD')
    # Note: Variation should have been variance
    plt.title(f"Moving Variation {base_name} - Normalized\n" 
              f"vector length{vector_length}, window size{window_size_var}, max Z score {threshold}, Number of datasets {n_rows}")
    plt.xlabel("Window Index")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)
    var_plot_name = f"{base_name}_var_n{vector_length}_w_mean{window_size_avg}_w_var{window_size_var}_t{threshold}_outlier_removed.png"
    plt.savefig(out_dir + var_plot_name, dpi=300)
    plt.close()

if __name__ == '__main__':
    tsv_file = str(sys.argv[1])
    pod_file = str(sys.argv[2])
    vector_length = int(sys.argv[3])
    window_size_avg = int(sys.argv[4])
    window_size_var = int(sys.argv[5])
    threshold = int(sys.argv[6])

    vectorize_and_plot(tsv_file, pod_file, vector_length, window_size_avg, window_size_var, threshold)