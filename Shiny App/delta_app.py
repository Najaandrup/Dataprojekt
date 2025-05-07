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


#def vectorize_and_plot(row, vector_length, window_size_avg, window_size_var, threshold, index=None):
#    read_id = row['read_id']
#    raw_signal = row['signal']
#    start = row['start']
#    end = row['end']
#    signal = raw_signal[start:end]
#    signal = remove_outliers(signal, threshold, index=index, original_start=start)
#
#    first_quantile = int(0.25 * len(signal))
#    third_quantile = int(0.75 * len(signal))
#    signal50 = signal[first_quantile:third_quantile]
#
#    if len(signal50) == 0 or np.mean(signal50) == 0:
#        return None, None
#
#    signal50_mean = np.mean(signal50)
#    normalized = signal / signal50_mean
#
#    means = []
#    variations = []
#
#    for j in range(vector_length):
#        x = int(j * (len(signal) / vector_length))
#
#        start_avg = max(int(x - window_size_avg // 2), 0)
#        end_avg = min(int(x + window_size_avg // 2), len(normalized))
#        window_avg = normalized[start_avg:end_avg]
#        means.append(np.mean(window_avg) if len(window_avg) > 0 else np.nan)
#
#        start_var = max(int(x - window_size_var // 2), 0)
#        end_var = min(int(x + window_size_var // 2), len(normalized))
#        window_var = normalized[start_var:end_var]
#        variations.append(np.var(window_var) if len(window_var) > 0 else np.nan)
#
#    x_vals = np.arange(vector_length)
#
#    fig, ax = plt.subplots(figsize=(10, 5))
#    ax.plot(x_vals, means, label='Mean', color='blue')
#    ax.plot(x_vals, variations, label='Variance', color='orange')
#    ax.set_title(f"Mean & Variance - Read {read_id}")
#    ax.set_xlabel("Window Index")
#    ax.set_ylabel("Value")
#    ax.grid(True)
#    ax.legend()
#
#    fig.tight_layout()
#
#    return fig

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
        #ui.input_slider("vector_size", "Vector size slider", 30, 960, value=30, step=30),
        ui.input_select("vector_size", "Vector size selector", choices=[30, 60, 120, 240, 480, 960]),
        ui.h5("Window sizes"),
        ui.input_selectize("avg_window_size", "Average Window Size", choices=[2, 5, 10, 20, 30, 60]),
        ui.input_selectize("var_window_size", "Variance Window Size", choices=[5, 10, 20, 30, 60, 120]),
        #ui.input_radio_buttons("z_score", "z-score", {"z-score 3": "3", "z-score 4": "4", "z-score 5": "5"}),
        ui.input_slider("z_score", "Z-score", 1, 6, 1),
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
    #title="Shiny app!"
)


# ---------------------------------------------------------------------
# Server Logic
# ---------------------------------------------------------------------

def server(input, output, session):
    
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
        

    #@render.plot
    #def combined_plot_mod():
    #    df = data_mod()
    #    if df is None or input.row_index() < 0 or input.row_index() >= len(df):
    #        print("No data or invalid row index.")
    #        return
#
    #    fig = vectorize_and_plot(
    #        df.iloc[input.row_index()],
    #        vector_length=int(input.vector_size()),
    #        window_size_avg=int(input.avg_window_size()),
    #        window_size_var=int(input.var_window_size()),
    #        threshold=float(input.z_score()),
    #        index=input.row_index()
    #    )
#
    #    if fig is None:
    #        print("Plot generation returned None.")
    #        return
#
    #    return fig
    #
    #@render.plot
    #def combined_plot_ctrl():
    #    df = data_ctrl()
    #    if df is None or input.row_index() < 0 or input.row_index() >= len(df):
    #        print("No data or invalid row index.")
    #        return
#
    #    fig = vectorize_and_plot(
    #        df.iloc[input.row_index()],
    #        vector_length=int(input.vector_size()),
    #        window_size_avg=int(input.avg_window_size()),
    #        window_size_var=int(input.var_window_size()),
    #        threshold=float(input.z_score()),
    #        index=input.row_index()
    #    )
#
    #    if fig is None:
    #        print("Plot generation returned None.")
    #        return
#
    #    return fig

    @render.plot
    def mean_plot():
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
        ax.legend()
        ax.grid(True)
        return fig


    @render.plot
    def variance_plot():
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
        ax.legend()
        ax.grid(True)
        return fig

    @render.plot
    def mean_plot():
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
        ax.plot(x_vals, mean_mod - mean_ctrl, color='blue')
        ax.set_title("Delta")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Delta Mean")
        ax.legend()
        ax.grid(True)
        return fig

# ---------------------------------------------------------------------
# Launch App
# ---------------------------------------------------------------------

app = App(app_ui, server)