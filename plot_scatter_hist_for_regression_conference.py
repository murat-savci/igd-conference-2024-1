import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker

from my_phd_functions import mask_nan_outliers


# Load your data 4km
# df = pd.read_pickle('insitudb_rrs_satbands6_final_total_cleaned_20240705_225251.pkl')

# load your data 1km
# filename = 'insitudb_rrs_satbands6_final_1KM_20240725_total_cleaned_20240725_025452.pkl'
filename = 'insitudb_rrs_satbands6_final_total_cleaned_20240705_225251.pkl'

# latest data tr0.0
# filename = 'insitudb_rrs_satbands6_final_total_cleaned_20240705_225251-tr0.0-ML-ready-20240810.pkl'
# latest data tr0.5
# filename = 'insitudb_rrs_satbands6_final_total_cleaned_20240705_225251-tr0.5-ML-ready-20240811.pkl'

df = pd.read_pickle(filename)

# Check the filename to determine the spatial resolution
if '1KM' in filename:
    spatial_resolution = 1
else:
    spatial_resolution = 4

# Set the threshold for outlier removal
threshold = 0.

def plot_scatter_plot(ax, insitu_band_val: pd.Series, sat_band_val: pd.Series, title: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Plot a scatter plot comparing two bands, fit a linear regression line, and display statistics.

    Parameters:
    - ax: The axis object to plot on
    - insitu_band_val: pd.Series, the in-situ band values
    - sat_band_val: pd.Series, the satellite band values
    - title: str, the title of the plot

    Returns:
    - None
    """
    insitu_band_val, sat_band_val, time = mask_nan_outliers(insitu_band_val, sat_band_val, threshold, debug=True)
   
    # Calculate the maximum value from both datasets for the plot
    max_value = max(max(insitu_band_val), max(sat_band_val))

    # Reshape the data to numpy array
    insitu_band_val_array = insitu_band_val.values.reshape(-1, 1)
    sat_band_val_array = sat_band_val.values.reshape(-1, 1)

    # split data into training and test sets
    insitu_train, insitu_test, sat_train, sat_test = train_test_split(insitu_band_val_array, sat_band_val_array, test_size=0.2, random_state=42)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(insitu_train, sat_train)
        
    # Make predictions
    
    sat_band_val_pred = model.predict(insitu_test)

    # Get the slope and intercept
    slope:float = model.coef_[0][0]
    intercept:float = model.intercept_[0]

    # Print the Total points, R2 score, RMSE, slope and intercept
    total_points = len(sat_test)
    r2 = r2_score(sat_test, sat_band_val_pred)
    rmse = np.sqrt(mean_squared_error(sat_test, sat_band_val_pred))
    mae = mean_absolute_error(sat_test, sat_band_val_pred)

    # Create a range of values for the regression line
    x_range = np.linspace(min(sat_test), max(sat_test), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    # Create a line plot 45 degree line with max value
    # x_line = np.linspace(0, max_value, 100)
    # y_line = x_line

    # ax.plot(x_line, y_line, color='black', linestyle='--', label='45 degree line')
    ax.plot(x_range, y_range, color='red', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
    ax.scatter(sat_test, sat_band_val_pred, alpha=0.5, color='orange', edgecolor='wheat', marker='o', s=17)
    ax.set_xlabel('Test data')
    ax.set_ylabel('Predicted data')
    ax.set_xlim(0, 0.009)
    ax.set_ylim(0, 0.009)
    # ax.set_xlim(0, 0.02)
    # ax.set_ylim(0, 0.02)
    # ax.set_xlim(0, np.round(max(max(sat_test), max(sat_band_val_pred)) * 1.05, 2))  # Adding a 5% buffer to the max value
    # ax.set_ylim(0, np.round(max(max(sat_test), max(sat_band_val_pred)) * 1.05, 2))  # Adding a 5% buffer to the max value
    
    # Format the tick labels to two decimal places
    formatter = ticker.FormatStrFormatter('%.3f')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_title(title)

    # Add statistics box
    stats_text = (f'Total points: {total_points}\n'
                  f'MAE: {mae:.4f}\n'
                  f'R2: {r2:.2f}\n'
                  f'RMSE: {rmse:.4f}\n'
                  f'Slope: {slope:.2f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.grid(True)

    return sat_band_val_pred, sat_test

def plot_histogram(ax, insitu_band_val: pd.Series, sat_band_val: pd.Series, title: str, sat_band_val_pred:np.ndarray, sat_test:np.ndarray) -> None:
    """
    Plot a histogram of the in-situ and satellite data together.

    Parameters:
    - ax: The axis object to plot on
    - insitu_band_val: pd.Series, the in-situ band values
    - sat_band_val: pd.Series, the satellite band values
    - title: str, the title of the plot

    Returns:
    - None
    """
    # filter out NAN values
    insitu_band_val, sat_band_val, time = mask_nan_outliers(insitu_band_val, sat_band_val, threshold)
    ax.hist(sat_test, bins=30, color='red', edgecolor='red', alpha=0.6, weights=np.ones(len(sat_test)) / len(sat_test) * 100, label=f'test data')
    ax.hist(sat_band_val_pred, bins=30, color='orange', edgecolor='orange', alpha=0.6, weights=np.ones(len(sat_band_val_pred)) / len(sat_band_val_pred) * 100, label=f'predicted data')
    ax.set_title(title)
    ax.set_xlabel('Reflectance Values')
    ax.set_ylabel('Percentage')
    ax.legend()
    ax.grid(True)

def plot_linear_regression_residuals(ax, insitu_band_val:pd.Series, sat_band_val: pd.Series, title: str) -> np.ndarray:
    """
    Plot the residuals of the linear regression model.

    Parameters:
    - sat_band_val: pd.Series, the satellite band values
    - sat_band_val_pred: np.ndarray, the predicted satellite band values
    - title: str, the title of the plot

    Returns:
    - None
    """
    insitu_band_val, sat_band_val, _ = mask_nan_outliers(insitu_band_val, sat_band_val, threshold)

    insitu_band_val_array = insitu_band_val.values.reshape(-1, 1)
    sat_band_val_array = sat_band_val.values.reshape(-1, 1)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(insitu_band_val_array, sat_band_val_array)

    # Make predictions
    sat_band_val_pred = model.predict(insitu_band_val_array)

    # Calculate the residuals
    residuals = sat_band_val_array - sat_band_val_pred

    # Plot the residuals
    ax.scatter(sat_band_val_array, residuals, alpha=0.5, color='orange', edgecolor='wheat', marker='o', s=17)
    ax.axhline(y=0, color='black', linestyle='--')
    ax.set_xlabel('Predicted values')
    ax.set_ylabel('Residuals')
    ax.set_title(title)
    ax.grid(True)

    return residuals

def plot_residual_histogram(ax, residuals: np.ndarray, title: str) -> None:
    """
    Plot a histogram of the residuals of the linear regression model.

    Parameters:
    - residuals: np.ndarray, the residuals of the linear regression model
    - title: str, the title of the plot

    Returns:
    - None
    """
    ax.hist(residuals, bins=100, color='orange', edgecolor='orange', alpha=0.6, weights=np.ones(len(residuals)) / len(residuals) * 100)
    ax.set_title(title)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Percentage')
    ax.set_xlim(-0.003, 0.003)  # Adding a 5% buffer to the max value
    # ax.set_xlim(-np.round(max(residuals) * 1.05, 2), np.round(max(residuals) * 1.05, 2))  # Adding a 5% buffer to the max value
    
    # Format the tick labels to two decimal places
    xformatter = ticker.FormatStrFormatter('%.3f')
    yformatter = ticker.FormatStrFormatter('%d')
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(yformatter)
    ax.grid(True)

# Example usage
def main():
    # Prepare a figure with A4 width and proportional height
    # fig, axd = plt.subplot_mosaic(
    #     [['scatter', 'histogram']],
    #     figsize=(8.27, 4.13)
    # )

    fig, axd = plt.subplot_mosaic(
        [['scatter', 'histogram'],
        ['residual', 'hist_residual']],
        figsize=(8.27, 8)
    )
 
 
    # add bottom right corner of the figure a title
    # fig.text(0.5, 0.011, f"Spatial resolution: {spatial_resolution}km\nFiltered outlier percent: {int(threshold * 100)}%", ha='center', fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.2'))

    fig.text(0.5, 0.011, f"Spatial resolution: {spatial_resolution}km", ha='center', fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.2'))
    # band       0          1          2          3          4          5
    bands = ['rrs_412', 'rrs_443', 'rrs_490', 'rrs_510', 'rrs_560', 'rrs_665']
    band = bands[5]

    bandname = band.replace('_', ' ').replace('rrs', 'Rrs')
    sat_band_val_pred, sat_test = plot_scatter_plot(axd['scatter'], df[f'{band}_insitu'], df[f'{band}_sat'], f'Scatter Plot of {bandname}')
    plot_histogram(axd['histogram'], df[f'{band}_insitu'], df[f'{band}_sat'], f'Histogram of {bandname}', sat_band_val_pred, sat_test)
    residuals = plot_linear_regression_residuals(axd['residual'], df[f'{band}_insitu'], df[f'{band}_sat'], f'Residuals of {bandname} band')
    plot_residual_histogram(axd['hist_residual'], residuals, f'Histogram of residuals of {bandname} band')
    plt.tight_layout()
    # plt.savefig(f'regression_model_{band}_scatter_histogram_residual_reshist_{spatial_resolution}km_{int(threshold * 100)}_percent.png', dpi=300)
    print(f"Saved regression_model_{band}_scatter_histogram_residual_reshist_{spatial_resolution}km_{int(threshold * 100)}_percent.png")
    plt.show()

if __name__ == '__main__':
    main()
