import os
import sys
import pandas as pd
import pod5 as p5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load_and_merge_data(tsv_file, pod_file):
    """Reads TSV and POD5 files, merging them into a DataFrame."""
    print(f"Reading TSV file: {tsv_file}")
    polyA_df = pd.read_csv(tsv_file, sep='\t')

    print(f"Reading POD5 file: {pod_file}")
    with p5.Reader(pod_file) as reader:
        pod_data = [(str(read.read_id), read.signal) for read in reader.reads()]

    pod_df = pd.DataFrame(pod_data, columns=["read_id", "signal"])
    merged_df = polyA_df.merge(pod_df, on="read_id")

    print(f"Merged DataFrame contains {len(merged_df)} rows.")
    return merged_df


def plot_polyA_signals_to_pdf(df, output_pdf, max_plots=60, max_length=11508):
    """Plots polyA signals and saves them to a PDF."""
    num_plots = min(len(df), max_plots)
    rows, cols = 5, 3  # Number of plots per page
    pages = (num_plots + (rows * cols) - 1) // (rows * cols)

    global_max_length = 0

    print(f"Generating PDF: {output_pdf}")

    with PdfPages(output_pdf) as pdf:  # ✅ Now correctly references the class
        for page in range(pages):
            fig, axes = plt.subplots(rows, cols, figsize=(15, 9))
            fig.suptitle(f'PolyA Signals (Page {page+1})', fontsize=16)
            axes = axes.flatten()

            for i in range(rows * cols):
                idx = page * (rows * cols) + i
                if idx >= num_plots:
                    axes[i].axis('off')
                    continue

                signal_slice = df['signal'][idx][df['start'][idx]:df['end'][idx]]
                max_length_sliced = len(signal_slice)
                global_max_length = max(global_max_length, max_length_sliced)

                axes[i].plot(signal_slice)
                axes[i].set_title(f'Row {idx+1}')
                axes[i].set_xlim(-100, max_length + 100)
                axes[i].set_ylim(-10, 1000)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"PDF saved as '{output_pdf}'")
    print(f"Max signal length: {global_max_length}")


# Ensure script follows cluster execution format
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python polyA_analysis.py <tsv_file> <pod5_file> <output_pdf>")
        sys.exit(1)

    tsv_file = sys.argv[1]
    pod5_file = sys.argv[2]
    output_pdf = sys.argv[3]

    # Optional arguments
    max_plots = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    max_length = int(sys.argv[5]) if len(sys.argv) > 5 else 11508

    if not os.path.exists(tsv_file):
        print(f"Error: TSV file '{tsv_file}' not found.")
        sys.exit(1)

    if not os.path.exists(pod5_file):
        print(f"Error: POD5 file '{pod5_file}' not found.")
        sys.exit(1)

    data = load_and_merge_data(tsv_file, pod5_file)
    plot_polyA_signals_to_pdf(data, output_pdf, max_plots, max_length)