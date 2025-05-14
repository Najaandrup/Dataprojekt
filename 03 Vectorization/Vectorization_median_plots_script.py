import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pod5 as p5

def process_and_plot_data(tsv_file, pod_file, nwindows):
    # Load data from TSV file
    polyA_df = pd.read_csv(tsv_file, sep='\t', skiprows=1, skipfooter=3, engine='python')
    
    # Load data from POD5 file
    with p5.Reader(pod_file) as reader:
        pod_data = [(str(read.read_id), read.signal, read.pore) for read in reader.reads()]
    pod_df = pd.DataFrame(pod_data, columns=["read_id","signal","pore"])
    
    # Merge data into a single DataFrame
    test_data = polyA_df.merge(pod_df, on="read_id")
    
    # Convert 'signal' column from string to list of numbers if necessary
    if isinstance(test_data['signal'].iloc[0], str):
        test_data['signal'] = test_data['signal'].apply(lambda x: np.array(list(map(float, x.strip('[]').split(',')))))
    
    # Extract polya_list with valid slices
    polya_list = [test_data['signal'][i][test_data['start'][i]:test_data['end'][i]] for i in range(len(test_data))]
    
    # Remove sequences shorter than nwindows
    polya_list = [seq for seq in polya_list if len(seq) >= nwindows]
    
    # Compute mean values for each window
    list_vectorized = []
    for vector_test in polya_list:
        window_size = len(vector_test) // nwindows
        lwin = np.repeat(window_size, nwindows)
        
        rest = len(vector_test) - window_size * nwindows
        add_positions = np.random.choice(nwindows, rest, replace=False)
        lwin[add_positions] += 1
        
        window_ends = np.cumsum(lwin)
        window_starts = window_ends - lwin + 1
        
        vector_mean = [np.mean(vector_test[start-1:end]) for start, end in zip(window_starts, window_ends)]
        list_vectorized.append(vector_mean)
    
    # Convert to matrix and normalize
    polya_matrix = np.array(list_vectorized)
    row_medians = np.median(polya_matrix, axis=1)
    polya_matrix_normalized = polya_matrix / row_medians[:, None]
    
    # Compute mean and standard deviation for each column
    col_means = np.mean(polya_matrix_normalized, axis=0)
    col_sd = np.std(polya_matrix_normalized, axis=0)
    
    # Plot results
    x = np.arange(1, nwindows + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x, col_means, color='blue', label='Mean', linewidth=2)
    plt.fill_between(x, col_means - col_sd, col_means + col_sd, color='lightblue', alpha=0.4, label='SD Range')
    
    plt.title('Mean ± Standard Deviation for Each Column')
    plt.xlabel('Column Index')
    plt.ylabel('Normalized Value')
    plt.legend()


    # Extract the base name of the pod5 file (before '_polya_reads')
    base_pod_filename = os.path.basename(pod_file).split('_polya_reads')[0]

    # Generate the output filename with window size included (e.g., "example_window_10_plot.png")
    output_filename = f"{base_pod_filename}_window_{nwindows}_plot.png"

    # Save the plot to the specified location
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    tsv_file = str(sys.argv[1])
    pod5_file = str(sys.argv[2])
    nwindows = int(sys.argv[3])
    
    process_and_plot_data(tsv_file, pod5_file, nwindows)
