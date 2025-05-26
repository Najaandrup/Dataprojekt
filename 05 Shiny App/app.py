from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pod5 as p5


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_data(tsv_file, pod_file):
    # Load data from tsv-file
    polyA_df = pd.read_csv(tsv_file, sep='\t', skiprows=1, skipfooter=3, engine='python')
    polyA_df.columns = polyA_df.columns.str.strip()  # Fix possible whitespace in headers

    # Load data from pod5-file
    with p5.Reader(pod_file) as reader:
        pod_data = {str(read.read_id): read.signal for read in reader.reads()}

    # Adds pod5 data to dataframe by read_ids
    polyA_df['signal'] = polyA_df['read_id'].map(pod_data)  # Adds NaN if read_id not found in pod data
    
    # Removes NaNs from dataframe if there is any
    polyA_df.dropna(subset=['signal'], inplace=True)

    return polyA_df


def remove_outliers(signal, threshold, return_mask=False):
    mean = np.mean(signal)
    std = np.std(signal)
    
    if std == 0:
        # If std is 0 nothing is an outlier and the signal will be returned
        # If return_mask = True, the signal and a boolean array of outlier status is returned
        return signal if not return_mask else (signal, np.full(len(signal), True))

    z_scores = np.abs((signal - mean) / std)
    mask = z_scores < threshold  # returns True if value if z-score is below the threshold
    filtered_signal = signal[mask]  # returns signal where mask is True (signal without outliers)

    # Returns filtered signal if return_mask = False
    # Returns filtered signal and a boolean array of outlier status for original data if return_mask = True
    return filtered_signal if not return_mask else (filtered_signal, mask)


def vectorize(row, vector_length, window_size_avg, window_size_var, threshold):
    raw_signal = row['signal']
    start = row['start']
    end = row['end']

    # Only takes signal of PolyA-tail
    signal = raw_signal[start:end]

    # Uses function above to remove outlies from PolyA-tail signal
    signal = remove_outliers(signal, threshold)

    # Computes 25-75% quantile region
    first_quantile = int(0.25 * len(signal))
    third_quantile = int(0.75 * len(signal))

    # Defines the signal in the 50% region
    signal50 = signal[first_quantile:third_quantile]

    # Computes the mean of signal50
    signal50_mean = np.mean(signal50)

    # Checks that signal50 is not empty or the mean of signal50 is not 0
    if len(signal50) == 0 or signal50_mean == 0:
        return None, None

    # Divides signal by the mean of signal50
    normalized = signal / signal50_mean

    # Empty lists for the means and variances
    means = []
    variances = []

    # Loop over lenght of the vector
    for j in range(vector_length):
        # Calculates evenly spaced position 'x' along the signal
        x = int(j * (len(signal) / vector_length))

        # Defines the start and end index of the window around position x
        start_avg = max(int(x - window_size_avg // 2), 0)
        end_avg = min(int(x + window_size_avg // 2), len(normalized))
        
        # Defines the window around x
        window_avg = normalized[start_avg:end_avg]
        
        # Appends the correct mean value of the window to the list 'means'
        means.append(np.mean(window_avg) if len(window_avg) > 0 else np.nan)

        # Repeats for the var values
        start_var = max(int(x - window_size_var // 2), 0)
        end_var = min(int(x + window_size_var // 2), len(normalized))
        
        window_var = normalized[start_var:end_var]
        
        # Appends the correct variance value of the window to the list 'variances'
        variances.append(np.var(window_var) if len(window_var) > 0 else np.nan)

    return means, variances  # returns list of means and variances


# ---------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("vector_size", "Vector size selector", choices=[30, 60, 120, 240, 480, 960], selected=60),
        ui.h5("Window sizes"),
        ui.input_select("avg_window_size", "Average - Window Size", choices=[2, 5, 10, 20, 30, 60], selected=60),
        ui.input_select("var_window_size", "Variance - Window Size", choices=[5, 10, 20, 30, 60, 120], selected=120),
        ui.input_slider("z_score", "Z-score", 1, 6, 3),
        ui.input_text("text", "Select Mod Row(s) (Use comma seperation)", "0,1"),  
        ui.input_numeric("row_index_ctrl", "Select Ctrl Row", value=0, min=0),
        ui.input_numeric("raw_mod_row", "Select Single Mod Row for Raw Signal Plot", value=0, min=0),
    ),
    ui.layout_columns(
        ui.panel_well(
            ui.h4("Upload Files With Modifications"),
            ui.input_file("pod5_file_mod", "Upload POD5 File", accept=[".pod5"]),
            ui.input_file("tsv_file_mod", "Upload TSV File", accept=[".tsv"]),
        ),
        ui.panel_well(
            ui.h4("Upload Control Files"),
            ui.input_file("pod5_file_ctrl", "Upload POD5 File", accept=[".pod5"]),
            ui.input_file("tsv_file_ctrl", "Upload TSV File", accept=[".tsv"]),
        )
    ),
    ui.output_plot("mean_plot"),
    ui.output_plot("variance_plot"),
    ui.output_plot("raw_signal_plot"),
    ui.output_plot("delta_mean_plot"),
    ui.output_plot("delta_var_plot")
)


# ---------------------------------------------------------------------
# Server Logic
# ---------------------------------------------------------------------

def server(input, output, session):
    
    def parse_indices(text):
        """Parse comma-separated string into list of ints, ignoring invalid entries."""
        if not text:
            return []
        return [int(x.strip()) for x in text.split(',') if x.strip().isdigit()]

    @reactive.Calc
    def all_files_uploaded():
        return (
            input.tsv_file_mod() and
            input.pod5_file_mod() and
            input.tsv_file_ctrl() and
            input.pod5_file_ctrl()
        )
    
    @reactive.Calc
    def data_mod():
        if not input.tsv_file_mod() or not input.pod5_file_mod():
            return None
        try:
            return load_data(input.tsv_file_mod()[0]["datapath"], input.pod5_file_mod()[0]["datapath"])
        except Exception as e:
            print(f"Error loading files: {e}")
            return None
    

    @reactive.Calc
    def data_ctrl():
        if not input.tsv_file_ctrl() or not input.pod5_file_ctrl():
            return None
        try:
            return load_data(input.tsv_file_ctrl()[0]["datapath"], input.pod5_file_ctrl()[0]["datapath"])
        except Exception as e:
            print(f"Error loading files: {e}")
            return None
        

    @render.plot
    def mean_plot():
        if not all_files_uploaded():
            print("Waiting for all files to be uploaded.")
            return
    
        df_mod = data_mod()
        df_ctrl = data_ctrl()

        if df_mod is None or df_ctrl is None:
            print("Missing data.")
            return
        
        indices_mod = parse_indices(input.text())
        idx_ctrl = int(input.row_index_ctrl())

        if not indices_mod:
            print("No mod row indices selected.")
            return
        
        if idx_ctrl < 0 or idx_ctrl >= len(df_ctrl):
            print("Invalid ctrl row index.")
            return
        
        mean_ctrl, _ = vectorize(
                df_ctrl.iloc[idx_ctrl],
                vector_length=int(input.vector_size()),
                window_size_avg=int(input.avg_window_size()),
                window_size_var=int(input.var_window_size()),
                threshold=float(input.z_score())
            )

        fig, ax = plt.subplots(figsize=(10, 5))

        for idx_mod in indices_mod:
            if idx_mod < 0 or idx_mod >= len(df_mod):
                continue

            mean_mod, _ = vectorize(
                df_mod.iloc[idx_mod],
                vector_length=int(input.vector_size()),
                window_size_avg=int(input.avg_window_size()),
                window_size_var=int(input.var_window_size()),
                threshold=float(input.z_score())
            )

            if mean_mod is not None:
                x_vals = np.arange(len(mean_mod))
                ax.plot(x_vals, mean_mod, label=f'Mod Mean {idx_mod}', alpha=0.7)
        
        if mean_ctrl is not None:
            x_vals = np.arange(len(mean_ctrl))
            ax.plot(x_vals, mean_ctrl, label='Ctrl Mean', color='black')

        ax.set_title("Mean Comparison")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Mean")
        ax.legend(loc='upper right')
        ax.grid(True)
        return fig


    @render.plot
    def variance_plot():
        if not all_files_uploaded():
            print("Waiting for all files to be uploaded.")
            return

        df_mod = data_mod()
        df_ctrl = data_ctrl()

        if df_mod is None or df_ctrl is None:
            print("Missing data.")
            return
        
        indices_mod = parse_indices(input.text())
        idx_ctrl = int(input.row_index_ctrl())

        if not indices_mod:
            print("No mod row indices selected.")
            return

        if idx_ctrl < 0 or idx_ctrl >= len(df_ctrl):
            print("Invalid ctrl row index.")
            return

    
        _, var_ctrl = vectorize(
            df_ctrl.iloc[idx_ctrl],
            vector_length=int(input.vector_size()),
            window_size_avg=int(input.avg_window_size()),
            window_size_var=int(input.var_window_size()),
            threshold=float(input.z_score())
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        for idx_mod in indices_mod:
            if idx_mod < 0 or idx_mod >= len(df_mod):
                continue

            _, var_mod = vectorize(
                df_mod.iloc[idx_mod],
                vector_length=int(input.vector_size()),
                window_size_avg=int(input.avg_window_size()),
                window_size_var=int(input.var_window_size()),
                threshold=float(input.z_score())
            )

            if var_mod is not None:
                x_vals = np.arange(len(var_mod))
                ax.plot(x_vals, var_mod, label=f'Mod Var {idx_mod}', alpha=0.7)
        
        if var_ctrl is not None:
            x_vals = np.arange(len(var_ctrl))
            ax.plot(x_vals, var_ctrl, label='Ctrl Var', color='black')

        ax.set_title("Variance Comparison")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Variance")
        ax.legend(loc='upper right')
        ax.grid(True)
        return fig


    @render.plot
    def raw_signal_plot():
        if not all_files_uploaded():
            print("Waiting for all files to be uploaded.")
            return

        df_mod = data_mod()
        df_ctrl = data_ctrl()

        if df_mod is None or df_ctrl is None:
            print("Missing data.")
            return

        idx_mod = int(input.raw_mod_row())
        idx_ctrl = int(input.row_index_ctrl())

        if (idx_mod < 0 or idx_mod >= len(df_mod) or idx_ctrl < 0 or idx_ctrl >= len(df_ctrl)):
            print("Index out of bounds.")
            return

        row_mod = df_mod.iloc[idx_mod]
        raw_signal_mod = row_mod['signal']
        start_mod = row_mod['start']
        end_mod = row_mod['end']
        signal_mod = raw_signal_mod[start_mod:end_mod]

        filtered_signal_mod, mask_mod = remove_outliers(signal_mod, float(input.z_score()), return_mask=True)

        indices_mod_signal = np.arange(start_mod, end_mod)
        filtered_indices_mod = indices_mod_signal[mask_mod]

        row_ctrl = df_ctrl.iloc[idx_ctrl]
        raw_signal_ctrl = row_ctrl['signal']
        start_ctrl = row_ctrl['start']
        end_ctrl = row_ctrl['end']
        signal_ctrl = raw_signal_ctrl[start_ctrl:end_ctrl]

        filtered_signal_ctrl, mask_ctrl = remove_outliers(signal_ctrl, float(input.z_score()), return_mask=True)

        indices_ctrl_signal = np.arange(start_ctrl, end_ctrl)
        filtered_indices_ctrl = indices_ctrl_signal[mask_ctrl]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        y_min = min(np.min(signal_mod), np.min(filtered_signal_mod),
                    np.min(signal_ctrl), np.min(filtered_signal_ctrl))
        y_max = max(np.max(signal_mod), np.max(filtered_signal_mod),
                    np.max(signal_ctrl), np.max(filtered_signal_ctrl))

        ax1.plot(indices_mod_signal, signal_mod, label='Mod raw', color='gray', alpha=0.7)
        ax1.plot(filtered_indices_mod, filtered_signal_mod, label='Mod filtered', color='red', alpha=0.5)
        ax1.set_title(f'Raw Signal Mod (Row {idx_mod})')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Current (pA)')
        ax1.set_ylim(y_min, y_max)
        ax1.grid(True)
        ax1.legend(loc='upper right')

        ax2.plot(indices_ctrl_signal, signal_ctrl, label='Ctrl raw', color='gray', alpha=0.7)
        ax2.plot(filtered_indices_ctrl, filtered_signal_ctrl, label='Ctrl filtered', color='blue', alpha=0.5)
        ax2.set_title(f'Raw Signal Ctrl (Row {idx_ctrl})')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Current (pA)')
        ax2.set_ylim(y_min, y_max)
        ax2.grid(True)
        ax2.legend(loc='upper right')

        fig.tight_layout()
        return fig


    @render.plot
    def delta_mean_plot():
        if not all_files_uploaded():
            print("Waiting for all files to be uploaded.")
            return

        df_mod = data_mod()
        df_ctrl = data_ctrl()

        if df_mod is None or df_ctrl is None:
            print("Missing data.")
            return
        
        indices_mod = parse_indices(input.text())
        idx_ctrl = int(input.row_index_ctrl())

        if not indices_mod:
            print("No mod row indices selected.")
            return

        if idx_ctrl < 0 or idx_ctrl >= len(df_ctrl):
            print("Invalid ctrl row index.")
            return

        _, mean_ctrl = vectorize(
            df_ctrl.iloc[idx_ctrl],
            vector_length=int(input.vector_size()),
            window_size_avg=int(input.avg_window_size()),
            window_size_var=int(input.var_window_size()),
            threshold=float(input.z_score())
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        for idx_mod in indices_mod:
            if idx_mod < 0 or idx_mod >= len(df_mod):
                continue

            _, mean_mod = vectorize(
                df_mod.iloc[idx_mod],
                vector_length=int(input.vector_size()),
                window_size_avg=int(input.avg_window_size()),
                window_size_var=int(input.var_window_size()),
                threshold=float(input.z_score())
            )

            if mean_mod is not None and mean_ctrl is not None:
                delta_mean = np.array(mean_mod) - np.array(mean_ctrl)
                x_vals = np.arange(len(delta_mean))
                ax.plot(x_vals, delta_mean, label=f'Delta Mean {idx_mod}', alpha=0.7)

        ax.set_title("Delta Mean (Mod - Ctrl)")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Delta Mean")
        ax.legend(loc='upper right')
        ax.grid(True)
        return fig


    @render.plot
    def delta_var_plot():
        if not all_files_uploaded():
            print("Waiting for all files to be uploaded.")
            return

        df_mod = data_mod()
        df_ctrl = data_ctrl()

        if df_mod is None or df_ctrl is None:
            print("Missing data.")
            return
        
        indices_mod = parse_indices(input.text())
        idx_ctrl = int(input.row_index_ctrl())

        if not indices_mod:
            print("No mod row indices selected.")
            return

        if idx_ctrl < 0 or idx_ctrl >= len(df_ctrl):
            print("Invalid ctrl row index.")
            return

        _, var_ctrl = vectorize(
            df_ctrl.iloc[idx_ctrl],
            vector_length=int(input.vector_size()),
            window_size_avg=int(input.avg_window_size()),
            window_size_var=int(input.var_window_size()),
            threshold=float(input.z_score())
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        for idx_mod in indices_mod:
            if idx_mod < 0 or idx_mod >= len(df_mod):
                continue

            _, var_mod = vectorize(
                df_mod.iloc[idx_mod],
                vector_length=int(input.vector_size()),
                window_size_avg=int(input.avg_window_size()),
                window_size_var=int(input.var_window_size()),
                threshold=float(input.z_score())
            )

            if var_mod is not None and var_ctrl is not None:
                delta_var = np.array(var_mod) - np.array(var_ctrl)
                x_vals = np.arange(len(delta_var))
                ax.plot(x_vals, delta_var, label=f'Delta Var {idx_mod}', alpha=0.7)

        ax.set_title("Delta Variance (Mod - Ctrl)")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Delta Variance")
        ax.legend(loc='upper right')
        ax.grid(True)
        return fig


app = App(app_ui, server)
