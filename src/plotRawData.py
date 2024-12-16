# The script plots data from CSV files in a 'data/raw' directory. It uses 
# pandas to read files, matplotlib and seaborn for plotting. It creates 
# subplots for each file, plotting numerical columns against the index. 
# The script sets titles, labels, and legends for each subplot, then 
# displays and saves the figure. It assumes CSV format with plottable 
# numerical data.


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_data_from_directory(data_dir):
    try:
        plt.style.use('ggplot')  # Using 'ggplot' style instead of 'seaborn'
    except:
        print("Warning: 'ggplot' style not available. Using default style.")

    sns.set_palette("deep")

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not files:
        print(f"No CSV files found in {data_dir}")
        return

    fig, axes = plt.subplots(len(files), 1, figsize=(
        12, 6*len(files)), squeeze=False)
    fig.suptitle('Data from raw files', fontsize=16)


    for i, file in enumerate(files):
        try:
            df = pd.read_csv(os.path.join(data_dir, file))

            # Ensure there are at least three columns
            if df.shape[1] < 3:
                print(
                    f"File {file} does not have at least three columns. Skipping.")
                continue

            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            color = df.iloc[:, 2]

            plt.figure(figsize=(12, 6))
            scatter = plt.scatter(x, y, c=color, cmap='viridis',
                                alpha=0.7, edgecolors='w', linewidth=0.5)
            plt.title(f'Data from {file}')
            plt.xlabel(df.columns[0])
            plt.ylabel(df.columns[1])
            cbar = plt.colorbar(scatter)
            cbar.set_label(df.columns[2])
            plt.tight_layout()
            plt.savefig(f'plots/{file}_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved for {file} as 'plots/{file}_scatter.png'.")
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

def main():
    data_dir = 'data/raw'
    plot_data_from_directory(data_dir)


if __name__ == "__main__":
    main()
