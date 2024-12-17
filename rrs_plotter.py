import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from typing import TypeVar, Optional, Union

from rrs_data import RRSData, RRSVariables, RRSBandName
from outlier_detector import OutlierDetector
from rrs_error_metrics import rrs_calculate_error_metrics, print_error_metrics_to_dataframe

# RRSData = TypeVar('RRSData')


def rrs_scatter_plot(insitu_band: pd.Series, sat_band: pd.Series, plot_param: Optional[plt.Axes] = None) -> Optional[plt.Axes]:
    """
    Creates a scatter plot of in-situ and satellite bands with a linear regression line and statistics box.

    Parameters:
    insitu_band (pd.Series): The in-situ band data.
    sat_band (pd.Series): The satellite band data.
    plot_param (plt.Axes or plt.Figure): The plot parameter, either an Axes object or a Figure object.

    Returns:
    plt.Axes or None: The Axes object if plot_param is an Axes object, otherwise None, and the plot is displayed.
    """
    # calculate max value of both bands
    insitu_band_max_val, sat_band_max_val = insitu_band.max(), sat_band.max()
    # Get the name of the bands before reshaping otherwise the name will be lost
    insitu_band_name, sat_band_name = insitu_band.name, sat_band.name
    # calculate error metrics before reshaping which will change the data type to numpy array
    error_metrics: dict = rrs_calculate_error_metrics(insitu_band, sat_band)
    # Reshape the data to numpy array: converts the 1D array into a 2D array with a single column
    insitu_band, sat_band = insitu_band.values.reshape(-1, 1), sat_band.values.reshape(-1,1)
    # Fit the linear regression model to see the trend of the data
    model = LinearRegression()
    model.fit(insitu_band, sat_band)
    # Get the slope and intercept
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    x_range = np.linspace(np.min(insitu_band), np.max(insitu_band), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    # Add statistics box to the plot
    stats_text = (
    f'N: {error_metrics["N"]}\n'
    f'R2: {error_metrics["R2"]}\n'
    f'MAE: {error_metrics["MAE"]}\n'
    f'RMSE: {error_metrics["RMSE"]}\n'
    # f'Mean bias: {error_metrics["Mean bias"]}\n'
    f'CV: {error_metrics["CV"]}'
    # f'Slope: {slope:.2f}\n'
    # f'Intercept: {intercept:.3f}'
    )

    # print(f"\nError Metrics of {insitu_band_name.replace('insitu', '').replace('_', ' ')}:\n{stats_text}\n")

    if plot_param is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        ax = plot_param

    ax.plot(x_range, y_range, color='red', label=f'{slope:.2f}x+{intercept:.2f}')
    ax.grid(True)
    ax.scatter(insitu_band, sat_band, alpha=0.5, color='orange', edgecolor='wheat', marker='o', s=17)
    insitu_band_name = insitu_band_name.replace('rrs', 'Rrs').replace('_', ' ')
    sat_band_name = sat_band_name.replace('_', ' ').replace('rrs', 'Rrs').replace('sat', 'satellite')
    title = insitu_band_name.replace('insitu', '')
    ax.set_title(title)
    ax.set_xlabel('In-situ Band')
    ax.set_ylabel('Satellite Band')
    ax.set_xlim(0, 0.03)
    ax.set_ylim(0, 0.03)
    ax.set_title(f'Scatter plot of {title}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    if plot_param is None:
        plt.show()
    
    return ax if plot_param is not None else None
        


def rrs_histogram_plot(insitu_band: pd.Series, sat_band: pd.Series, plot_param: Optional[plt.Axes] = None) -> Optional[plt.Axes]:
    """
    Creates a histogram plot comparing the distribution of in-situ and satellite RRS data.

    Parameters:
        insitu_band (pd.Series): The in-situ RRS data to be plotted.
        sat_band (pd.Series): The satellite RRS data to be plotted.
        plot_param (Union[plt.Axes, plt.Figure]): The plot parameter to be used for plotting. Defaults to plt.Figure().

    Returns:
        Optional[plt.Axes | None]: The axes of the plot if plot_param is plt.Axes, otherwise None.
    """
    if plot_param is None:
        fig, ax = plt.subplots()
    else:
        ax = plot_param

    ax.hist(insitu_band, bins=30, color='red', edgecolor='red', alpha=0.6, weights=np.ones(len(insitu_band)) / len(insitu_band) * 100, label=f'in-situ data')
    ax.hist(sat_band, bins=30, color='orange', edgecolor='orange', alpha=0.6, weights=np.ones(len(sat_band)) / len(sat_band) * 100, label=f'satellite data')
    ax.set_title(f"Histogram of {insitu_band.name.replace('_', ' ').replace('rrs', 'Rrs').replace('insitu', '')}")
    ax.set_xlabel(insitu_band.name.replace('_', ' ').replace('rrs', 'Rrs'))
    ax.set_ylabel('Percentage')
    ax.legend()
    ax.grid(True)

    if plot_param is None:
        plt.show()
    
    return ax if plot_param is not None else None
    

def rrs_time_series_plot(
    insitu_band: pd.Series,
    sat_band: pd.Series,
    time: pd.Series,
    plot_param: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Creates a time series plot comparing the in-situ and satellite RRS data.

    Parameters:
        insitu_band (pd.Series): The in-situ RRS data to be plotted.
        sat_band (pd.Series): The satellite RRS data to be plotted.
        time (pd.Series): The time data.
        plot_param (Optional[plt.Axes]): The plot parameter to be used for plotting. Defaults to None.

    Returns:
        Optional[plt.Axes]: The axes of the plot if plot_param is plt.Axes, otherwise None.
    """
    if plot_param is None:
        fig, ax = plt.subplots()
    else:
        ax = plot_param

    title = f"Time Series Analysis of {insitu_band.name.replace('insitu', '').replace('_', ' ').replace('rrs', 'Rrs')} band"
    ax.plot(time, insitu_band, label='In situ', color='red')
    ax.plot(time, sat_band, label='Sat band', color='orange')
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Reflectance Value')
    ax.legend()
    ax.grid(True)    

    if plot_param is None:
        plt.show()
    
    return ax if plot_param is not None else None
    

def main():
    filename = "insitudb_rrs_satbands6_final_total_cleaned_20240705_225251.pkl"
    rrs_data: RRSData = RRSData(filename)
    # insitu_band, sat_band = rrs_data.get_rrs_pair(RRSBandName.INSITU_BAND_412, RRSBandName.SAT_BAND_412)
    # time = rrs_data.time

    # print(insitu_band.info())
    # print()
    # print(sat_band.info())

    outlier_detector = OutlierDetector(rrs_data)

    # insitu_band, sat_band = outlier_detector.mask_nan_values(RRSBandName.INSITU_BAND_412, RRSBandName.SAT_BAND_412, debug=True)
    insitu_band, sat_band, time = outlier_detector.mask_nan_values(RRSBandName.INSITU_BAND_412, RRSBandName.SAT_BAND_412, time=RRSVariables.TIME, debug=True)


    # insitu_band, sat_band, time = outlier_detector.remove_outliers_difference_std(insitu_band, sat_band, time=time, threshold=2.2, debug=True)
    insitu_band, sat_band, time = outlier_detector.remove_outliers_by_threshold(insitu_band, sat_band, time=time, threshold=0.5, debug=True)

    print(len(insitu_band), len(sat_band), len(time))
   
    error_metrics: dict = rrs_calculate_error_metrics(insitu_band, sat_band)
    print(print_error_metrics_to_dataframe(error_metrics))
    
    # Prepare a figure with A4 width and proportional height
    fig, axd = plt.subplot_mosaic(
        [['scatter', 'histogram'],
         ['timeseries', 'timeseries']],
        figsize=(8.27, 8)
    )
    rrs_scatter_plot(insitu_band, sat_band, plot_param=axd['scatter'])
    rrs_histogram_plot(insitu_band, sat_band, plot_param=axd['histogram'])
    rrs_time_series_plot(insitu_band, sat_band, time, plot_param=axd['timeseries'])
    plt.tight_layout()
    plt.show()

    # rrs_scatter_plot(insitu_band, sat_band)
    # rrs_histogram_plot(insitu_band, sat_band)
    # rrs_time_series_plot(insitu_band, sat_band, time)

    # plt.hist(insitu_band, bins=30, color='red', edgecolor='red', alpha=0.6, weights=np.ones(len(insitu_band)) / len(insitu_band) * 100, label=f'in-situ data')

    

if __name__ == '__main__':
    main()