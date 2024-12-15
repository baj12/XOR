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

            ax = axes[i, 0]
            for column in df.select_dtypes(include=['float64', 'int64']):
                ax.plot(df.index, df[column], label=column)

            ax.set_title(f'Data from {file}')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.legend()
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

    plt.tight_layout()
    plt.show()
    plt.savefig('data_plot.png', dpi=300, bbox_inches='tight')


def main():
    data_dir = 'data/raw'
    plot_data_from_directory(data_dir)


if __name__ == "__main__":
    main()
