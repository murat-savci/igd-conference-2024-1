import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from typing import TypeVar, Optional, Union

from rrs_data import RRSData, RRSVariables, RRSBandName
from outlier_detector import OutlierDetector

# RRSData = TypeVar('RRSData')

filename = ''

# Check the filename to determine the spatial resolution
if '1KM' in filename:
    spatial_resolution = 1
else:
    spatial_resolution = 4

class RRSStats:
    pass


def calculate_error_metrics(insitu_band: pd.Series, sat_band: pd.Series) -> dict:
    pass

class RRSPlot:
    
    def __init__(self, rrs_data: 'RRSData'):
        self.rrs_data = rrs_data

    def rrs_scatter_plot(self, insitu_band: pd.Series, sat_band: pd.Series, title: str, regression_line: Optional[bool] = True) -> None:

        # calculate max value of both bands
        insitu_band_max_val, sat_band_max_val = insitu_band.max(), sat_band.max()
        # print(insitu_band_max_val, sat_band_max_val)
        #
        # Get the name of the bands before reshaping otherwise the name will be lost
        insitu_band_name, sat_band_name = insitu_band.name, sat_band.name
        # print(f'insitu_band.name is {insitu_band.name}, sat_band.name is {sat_band.name}')
        # print(f"shape of insitu band {insitu_band.shape}")
        #
        # Reshape the data to numpy array: converts the 1D array into a 2D array with a single column
        insitu_band, sat_band = insitu_band.values.reshape(-1, 1), sat_band.values.reshape(-1,1)
        # print(insitu_band.shape, type(insitu_band))
        # print(sat_band.shape, type(sat_band))

        if regression_line:
        # Fit the linear regression model to see the trend of the data
            model = LinearRegression()
            model.fit(insitu_band, sat_band)
            # Get the slope and intercept
            slope = model.coef_[0][0]
            intercept = model.intercept_[0]

        x_range = np.linspace(np.min(insitu_band), np.max(insitu_band), 100)
        y_range = slope * x_range + intercept
        y_range_new = model.predict(x_range.reshape(-1, 1))

        if y_range.all() == y_range_new.all():
            print('true')



def main():
    filename = "insitudb_rrs_satbands6_final_total_cleaned_20240705_225251.pkl"
    rrs_data: RRSData = RRSData(filename)
    insitu_band, sat_band = rrs_data.get_rrs_pair(RRSBandName.INSITU_BAND_412, RRSBandName.SAT_BAND_412)

    # print(insitu_band.info())
    # print()
    # print(sat_band.info())

    outlier_detector = OutlierDetector(rrs_data)

    insitu_band, sat_band = outlier_detector.mask_nan_values(RRSBandName.INSITU_BAND_412, RRSBandName.SAT_BAND_412, debug=True)

    outlier_detector.remove_outliers_difference_std(RRSBandName.INSITU_BAND_412, RRSBandName.SAT_BAND_412, debug=True)

    rrs_plotter = RRSPlot(rrs_data)
    rrs_plotter.rrs_scatter_plot(insitu_band, sat_band, title='test')


if __name__ == '__main__':
    main()