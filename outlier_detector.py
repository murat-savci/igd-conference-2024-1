import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple, Optional, Union, TypeVar
import collections.abc as collections # this is a replacement for typing module in python 3.9

from rrs_data import RRSData, RRSBandName, RRSVariables
# from rrs_data import RRSData, RRSBandName
from rrs_data_debuger import *


# type RRSData using collections.abc
# RRSData = TypeVar('RRSData')
# RRSBandName = TypeVar('RRSBandName')
# RRSVariables = TypeVar('RRSVariables')

class OutlierDetectionMethod(Enum):
    THRESHOLD = 'threshold'
    STD = 'std'
    IQR = 'iqr'
    ZSCORE = 'zscore'
    MAD = 'mad'


class OutlierDetector():
    def __init__(self, rrs_data: RRSData):
        self.rrs_data = rrs_data

    @mask_nan_values_debugger
    def mask_nan_values(
        self: 'OutlierDetector',
        insitu_band: RRSBandName,
        sat_band: RRSBandName,
        debug: Optional[bool] = False,
        time: Optional[RRSVariables] = None
    ) -> Union[Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Masks out NaN values from the given in-situ and satellite band values.

        Parameters:
            self (OutlierDetector): The OutlierDetector object.
            insitu_band (RRSBandName): The in-situ band value to be filtered.
            sat_band (RRSBandName): The satellite band value to be filtered.
            debug (Optional[bool]): A flag for printing debug information. Defaults to False.
            time (Optional[RRSVariables]): The time data. Defaults to None.

        Returns:
            Union[Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series, pd.Series]]: 
            A tuple containing the filtered in-situ band values, the filtered satellite band values, 
            and the filtered time data (if provided).
        """

        # print(f'time = {time}')
        # print(f'debug = {debug}')

        insitu_band: pd.Series = self.rrs_data.get_rrs_band(insitu_band)
        sat_band: pd.Series  = self.rrs_data.get_rrs_band(sat_band)
        # time: Optional[pd.Series] = self.rrs_data.time if time is not None else time
        mask: bool = insitu_band.notna() & sat_band.notna()
        # mask: bool = ~insitu_band.isna() & ~sat_band.isna()
        if time is None:
            return insitu_band[mask], sat_band[mask]
        else:
            time: pd.Series = self.rrs_data.time
            return insitu_band[mask], sat_band[mask], time[mask]
      
    @remove_outliers_debugger  
    def remove_outliers_by_threshold(
        self: 'OutlierDetector',
        insitu_band: pd.Series,
        sat_band: pd.Series,
        threshold: float = 0.0,
        time: Optional[pd.Series] = None,
        debug: bool = False
    ) -> Union[Tuple[pd.Series, pd.Series, pd.Series], Tuple[pd.Series, pd.Series]]:
        """
        Removes outliers from the given in-situ and satellite band values. This code is primarily created to use output of mask_nan_values method.
        First run mask_nan_values method the give output of the method to this method.

        Parameters:
            self (RRSData): The RRSData object.
            insitu_band (pd.Series): The in-situ band values.
            sat_band (pd.Series): The satellite band values.
            threshold (float, optional): The threshold value for outlier detection. Defaults to 0.5.
            time (Optional[pd.Series], optional): The time data. Defaults to None.
            debug (bool, optional): Whether to print debug information. Defaults to False.

        Returns:
            Union[Tuple[pd.Series, pd.Series, pd.Series], Tuple[pd.Series, pd.Series]]:
            A tuple containing the filtered in-situ band values, the filtered satellite band values, and the filtered time data (if provided).
        """
        mask = (sat_band > insitu_band * (1 + threshold)) | (sat_band < insitu_band * (1 - threshold))

        if time is not None:
            return insitu_band[~mask], sat_band[~mask], time[~mask]
        else:
            return insitu_band[~mask], sat_band[~mask]
        
    def remove_outliers_difference_std(
        self: OutlierDetector,
        insitu_band: pd.Series,
        sat_band: pd.Series,
        threshold: float = 1.5,
        time: Optional[pd.Series] = None,
        debug: bool = False
    ) -> Tuple[pd.Series, pd.Series, Optional[pd.Series]]:
        """
        Removes outliers in the difference between in-situ and satellite band values using the standard deviation method.

        Parameters:
            self (OutlierDetector): The OutlierDetector object.
            insitu_band (pd.Series): The in-situ band values.
            sat_band (pd.Series): The satellite band values.
            threshold (float, optional): The threshold value for outlier detection. Defaults to 1.5.
            time (Optional[pd.Series], optional): The time data. Defaults to None.
            debug (bool, optional): Whether to print debug information. Defaults to False.

        Returns:
            Tuple[pd.Series, pd.Series, Optional[pd.Series]]: A tuple containing the filtered in-situ band values, 
            the filtered satellite band values, and the filtered time data (if provided).
        """
        diff = insitu_band - sat_band
        mean = diff.mean()
        std = diff.std()

        if debug:
            print(f'\ninsitu_band_val: {insitu_band.name}')
            print(f"Number of data points before filtering: {insitu_band.count()}")
            print(f"Mean: {mean}")
            print(f"Standard Deviation: {std}")
        
        mask = (diff < (mean - std * threshold)) | (diff > (mean + std * threshold))
        insitu_band = insitu_band[~mask]
        sat_band = sat_band[~mask]

        if debug:
            no_of_filtered_data_points = insitu_band.count() # Number of data points after filtering
            print(f"Number of data points after removing outliers: {no_of_filtered_data_points}")

        if time is not None:
            time = time[~mask]
            return insitu_band, sat_band, time
        else:
            return insitu_band, sat_band

    def remove_outliers_diff_iqr(self: RRSData,
                                 insitu_band: pd.Series,
                                 sat_band: pd.Series,
                                 threshold: float = 1.5,
                                 debug: bool = False
                                 ) -> Tuple[pd.Series, pd.Series]:
        
        diff = insitu_band -sat_band
        Q1 = diff.quantile(0.25)
        Q3 = diff.quantile(0.75)
        IQR = Q3 - Q1

        if debug:
            print(f"Q1: {Q1}")
            print(f"Q3: {Q3}")
            print(f"IQR: {IQR}")
        


from rrs_data import RRSData, RRSVariables, RRSBandName
from rrs_error_metrics import rrs_calculate_error_metrics

def main():
    filename = "insitudb_rrs_satbands6_final_total_cleaned_20240705_225251.pkl"
    rrs_data= RRSData(filename)

    outlier_detector = OutlierDetector(rrs_data)
    rrs_412_insitu, rrs_412_sat, rrs_412_time = outlier_detector.mask_nan_values(RRSBandName.INSITU_BAND_412, RRSBandName.SAT_BAND_412, debug=True, time=RRSVariables.TIME)

    rrs_412_error_metrics = rrs_calculate_error_metrics(rrs_412_insitu, rrs_412_sat)
    print(rrs_412_error_metrics)

    outlier_detector.remove_outliers_diff_iqr(rrs_412_insitu, rrs_412_sat, debug=True)




if __name__ == '__main__':
    main()