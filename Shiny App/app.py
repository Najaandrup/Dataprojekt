from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pod5 as p5


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_data(tsv_file, pod_file):
    polyA_df = pd.read_csv(tsv_file, sep='\t', skiprows=1, skipfooter=3, engine='python')
    polyA_df.columns = polyA_df.columns.str.strip()

    with p5.Reader(pod_file) as reader:
        pod_data = {str(read.read_id): read.signal for read in reader.reads()}

    polyA_df['signal'] = polyA_df['read_id'].map(pod_data)
    polyA_df.dropna(subset=['signal'], inplace=True)

    return polyA_df


def remove_outliers(signal, threshold, index=None, return_mask=False, original_start=0):
    mean = np.mean(signal)
    std = np.std(signal)
    
    if std == 0:
        if index is not None:
            print(f"Row {index}: Standard deviation is 0, no outliers removed.")
        return signal if not return_mask else (signal, np.full(len(signal), True))

    z_scores = np.abs((signal - mean) / std)
    mask = z_scores < threshold
    filtered_signal = signal[mask]

    return filtered_signal if not return_mask else (filtered_signal, mask)



def vectorize(row, vector_length, window_size_avg, window_size_var, threshold, index=None):
    raw_signal = row['signal']
    start = row['start']
    end = row['end']
    signal = raw_signal[start:end]
    signal = remove_outliers(signal, threshold, index=index, original_start=start)

    first_quantile = int(0.25 * len(signal))
    third_quantile = int(0.75 * len(signal))
    signal50 = signal[first_quantile:third_quantile]

    if len(signal50) == 0 or np.mean(signal50) == 0:
        return None, None

    signal50_mean = np.mean(signal50)
    normalized = signal / signal50_mean

    means = []
    variations = []

    for j in range(vector_length):
        x = int(j * (len(signal) / vector_length))

        start_avg = max(int(x - window_size_avg // 2), 0)
        end_avg = min(int(x + window_size_avg // 2), len(normalized))
        window_avg = normalized[start_avg:end_avg]
        means.append(np.mean(window_avg) if len(window_avg) > 0 else np.nan)

        start_var = max(int(x - window_size_var // 2), 0)
        end_var = min(int(x + window_size_var // 2), len(normalized))
        window_var = normalized[start_var:end_var]
        variations.append(np.var(window_var) if len(window_var) > 0 else np.nan)

    return means, variations


# ---------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("vector_size", "Vector size selector", choices=[30, 60, 120, 240, 480, 960], selected=60),
        ui.h5("Window sizes"),
        ui.input_select("avg_window_size", "Average Window Size", choices=[2, 5, 10, 20, 30, 60], selected=60),
        ui.input_select("var_window_size", "Variance Window Size", choices=[5, 10, 20, 30, 60, 120], selected=120),
        ui.input_slider("z_score", "Z-score", 1, 6, 3),
        ui.input_numeric("row_index", "Row index", value=0, min=0)
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
    ui.output_plot("delta_var_plot"),
    #title="Shiny app!"
)


# ---------------------------------------------------------------------
# Server Logic
# ---------------------------------------------------------------------

def server(input, output, session):
    
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
        idx = input.row_index()

        if df_mod is None or df_ctrl is None:
            print("Missing data.")
            return

        if idx < 0 or idx >= len(df_mod) or idx >= len(df_ctrl):
            print("Invalid row index.")
            return

        mean_mod, _ = vectorize(
            df_mod.iloc[idx],
            vector_length=int(input.vector_size()),
            window_size_avg=int(input.avg_window_size()),
            window_size_var=int(input.var_window_size()),
            threshold=float(input.z_score()),
            index=idx
        )

        mean_ctrl, _ = vectorize(
            df_ctrl.iloc[idx],
            vector_length=int(input.vector_size()),
            window_size_avg=int(input.avg_window_size()),
            window_size_var=int(input.var_window_size()),
            threshold=float(input.z_score()),
            index=idx
        )

        if mean_mod is None or mean_ctrl is None:
            print("Failed to compute mean vectors.")
            return

        x_vals = np.arange(len(mean_mod))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_vals, mean_mod, label='Mod Mean', color='blue')
        ax.plot(x_vals, mean_ctrl, label='Ctrl Mean', color='green')
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
        idx = input.row_index()

        if df_mod is None or df_ctrl is None:
            print("Missing data.")
            return

        if idx < 0 or idx >= len(df_mod) or idx >= len(df_ctrl):
            print("Invalid row index.")
            return

        _, var_mod = vectorize(
            df_mod.iloc[idx],
            vector_length=int(input.vector_size()),
            window_size_avg=int(input.avg_window_size()),
            window_size_var=int(input.var_window_size()),
            threshold=float(input.z_score()),
            index=idx
        )

        _, var_ctrl = vectorize(
            df_ctrl.iloc[idx],
            vector_length=int(input.vector_size()),
            window_size_avg=int(input.avg_window_size()),
            window_size_var=int(input.var_window_size()),
            threshold=float(input.z_score()),
            index=idx
        )

        if var_mod is None or var_ctrl is None:
            print("Failed to compute variance vectors.")
            return

        x_vals = np.arange(len(var_mod))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_vals, var_mod, label='Mod Variance', color='orange')
        ax.plot(x_vals, var_ctrl, label='Ctrl Variance', color='red')
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
        idx = input.row_index()

        if (
            df_mod is None or df_ctrl is None or
            idx < 0 or idx >= len(df_mod) or idx >= len(df_ctrl)
        ):
            print("Invalid or missing data.")
            return

        # Modification data
        row_mod = df_mod.iloc[idx]
        raw_signal_mod = row_mod['signal']
        start_mod = row_mod['start']
        end_mod = row_mod['end']
        signal_mod = raw_signal_mod[start_mod:end_mod]

        filtered_signal_mod, mask_mod = remove_outliers(signal_mod, float(input.z_score()), index=idx, return_mask=True, original_start=start_mod)

        indices_mod = np.arange(start_mod, end_mod)
        filtered_indices_mod = indices_mod[mask_mod]

        # Control data
        row_ctrl = df_ctrl.iloc[idx]
        raw_signal_ctrl = row_ctrl['signal']
        start_ctrl = row_ctrl['start']
        end_ctrl = row_ctrl['end']
        signal_ctrl = raw_signal_ctrl[start_ctrl:end_ctrl]

        filtered_signal_ctrl, mask_ctrl = remove_outliers(signal_ctrl, float(input.z_score()), index=idx, return_mask=True, original_start=start_ctrl)

        indices_ctrl = np.arange(start_ctrl, end_ctrl)
        filtered_indices_ctrl = indices_ctrl[mask_ctrl]

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        y_min = min(np.min(signal_mod), np.min(filtered_signal_mod),
                    np.min(signal_ctrl), np.min(filtered_signal_ctrl))
        y_max = max(np.max(signal_mod), np.max(filtered_signal_mod),
                    np.max(signal_ctrl), np.max(filtered_signal_ctrl))

        ax1.plot(indices_mod, signal_mod, label='Mod raw', color='gray')
        ax1.plot(filtered_indices_mod, filtered_signal_mod, label='Mod filtered', color='red', alpha=0.7)
        ax1.set_title('Raw Signal - MOD')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Signal')
        ax1.set_ylim(y_min, y_max)
        ax1.grid(True)
        ax1.legend(loc='upper right')

        ax2.plot(indices_ctrl, signal_ctrl, label='Ctrl raw', color='gray')
        ax2.plot(filtered_indices_ctrl, filtered_signal_ctrl, label='Ctrl filtered', color='blue', alpha=0.7)
        ax2.set_title('Raw Signal - CTRL')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Signal')
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
        idx = input.row_index()

        if df_mod is None or df_ctrl is None:
            print("Missing data.")
            return

        if idx < 0 or idx >= len(df_mod) or idx >= len(df_ctrl):
            print("Invalid row index.")
            return

        mean_mod, _ = vectorize(
            df_mod.iloc[idx],
            vector_length=int(input.vector_size()),
            window_size_avg=int(input.avg_window_size()),
            window_size_var=int(input.var_window_size()),
            threshold=float(input.z_score()),
            index=idx
        )

        mean_ctrl, _ = vectorize(
            df_ctrl.iloc[idx],
            vector_length=int(input.vector_size()),
            window_size_avg=int(input.avg_window_size()),
            window_size_var=int(input.var_window_size()),
            threshold=float(input.z_score()),
            index=idx
        )

        if mean_mod is None or mean_ctrl is None:
            print("Failed to compute mean vectors.")
            return

        x_vals = np.arange(len(mean_mod))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_vals, np.array(mean_mod) - np.array(mean_ctrl), color='blue')
        ax.set_title("Delta")
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
        idx = input.row_index()

        if df_mod is None or df_ctrl is None:
            print("Missing data.")
            return

        if idx < 0 or idx >= len(df_mod) or idx >= len(df_ctrl):
            print("Invalid row index.")
            return

        _, var_mod = vectorize(
            df_mod.iloc[idx],
            vector_length=int(input.vector_size()),
            window_size_avg=int(input.avg_window_size()),
            window_size_var=int(input.var_window_size()),
            threshold=float(input.z_score()),
            index=idx
        )

        _, var_ctrl = vectorize(
            df_ctrl.iloc[idx],
            vector_length=int(input.vector_size()),
            window_size_avg=int(input.avg_window_size()),
            window_size_var=int(input.var_window_size()),
            threshold=float(input.z_score()),
            index=idx
        )

        if var_mod is None or var_ctrl is None:
            print("Failed to compute variance vectors.")
            return

        x_vals = np.arange(len(var_mod))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_vals, np.array(var_mod) - np.array(var_ctrl), color='blue')
        ax.set_title("Delta")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Delta Variance")
        ax.legend(loc='upper right')
        ax.grid(True)
        return fig



# ---------------------------------------------------------------------
# Launch App
# ---------------------------------------------------------------------

app = App(app_ui, server)