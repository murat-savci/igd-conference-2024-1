# Decorator functions for debugging
import pandas as pd
from typing import Tuple, Optional, TypeVar, Callable

RRSData = TypeVar('RRSData')
RRSBandName = TypeVar('RRSBandName')
RRSVariables = TypeVar('RRSVariables')
OutlierDetector = TypeVar('OutlierDetector')



# Decorator function for debugging the mask_nan_outliers method in the RrsData class
def mask_nan_values_debugger(mask_nan_outliers: Callable):
    """
    A decorator function for debugging the mask_nan_outliers method in the RrsData class.
    
    Parameters
    ----------
    mask_nan_outliers : Callable[[RRSData, RRSBandName, RRSBandName, Optional[bool], Optional[pd.Series]], Tuple[pd.Series, pd.Series, Optional[pd.Series]]]
        The mask_nan_outliers method to be decorated.
    
    Returns
    -------
    Callable[[RRSData, RRSBandName, RRSBandName, Optional[bool], Optional[pd.Series]], Tuple[pd.Series, pd.Series, Optional[pd.Series]]]
        The decorated mask_nan_outliers method.
    """
    def wrapper(
        self: 'OutlierDetector',
        insitu_band: 'RRSBandName', 
        sat_band: 'RRSBandName', 
        debug: Optional[bool] = False, 
        time: Optional[RRSVariables] = None
    ) -> Tuple[pd.Series, pd.Series, Optional[pd.Series]]:
        """
        A decorator function for debugging the mask_nan_outliers method in the RrsData class.
        
        Parameters
        ----------
        self : RRSData
            The RRSData object containing the in-situ and satellite band values.
        insitu_band_num : RRSBandName
            The in-situ band value to be filtered.
        sat_band_num : RRSBandName
            The satellite band value to be filtered.
        debug : Optional[bool]
            A flag for printing debug information. Defaults to False.
        time : Optional[pd.Series]
            The time data. Defaults to None.
        
        Returns
        -------
        Tuple[pd.Series, pd.Series, Optional[pd.Series]]
            The filtered in-situ and satellite band values.
        """
        if debug:
            dashes = "-" * 40
            insitu_band_count: pd.Series = self.rrs_data.get_rrs_band(insitu_band).count()
            sat_band_count: pd.Series = self.rrs_data.get_rrs_band(sat_band).count()
            print(f'\n----------------------------------------------------------------------------')
            print(f'\tBegining of mask nan values debugger:')
            print(f'----------------------------------------------------------------------------\n')
            print(f"\n{dashes}\tband {insitu_band.value}\t{dashes}\n")
            print(f"Number of insitu data points before filtering: {insitu_band_count}")
            print(f"Number of sat data points before filtering: {sat_band_count}")
            if time is not None:
                time_count: pd.Series = self.rrs_data.time.count()
                print(f"Number of time data points before NAN filtering: {time_count}")
            else:
                print(f'Time parameter is not set.')
            print(f'\n----------------------------------------------------------------------------\n')

        result = mask_nan_outliers(self, insitu_band, sat_band, debug, time)
        
        if debug:
            if time is not None:
                insitu_band, sat_band, time = result
                print(f"Number of insitu data points after NAN filtering: {insitu_band.count()}")
                print(f"Number of sat data points after NAN filtering: {sat_band.count()}")
                print(f"Number of time data points after NAN filtering: {time.count()}")
            else:
                insitu_band, sat_band = result
                print(f"Number of insitu data points after NAN filtering: {insitu_band.count()}")
                print(f"Number of sat data points after NAN filtering: {sat_band.count()}")
            print(f'\n----------------------------------------------------------------------------')
            print(f'\tEnding of mask nan values debugger:')
            print(f'----------------------------------------------------------------------------\n')
        return result
    return wrapper


def remove_outliers_debugger(remove_outliers: callable):
    def wrapper(self: 'RRSData', *args, **kwargs):
        debug: bool = kwargs.get('debug', False)
        insitu_band, sat_band = args[:2]
        time: Optional[pd.Series] = kwargs.get('time')
        threshold: float = kwargs.get('threshold', 0.0)
        if debug:
            print(f'\n----------------------------------------------------------------------------')
            print(f'\tBegining of Remove outliers method debugger: Filtering threshold: {threshold}%')
            print(f'----------------------------------------------------------------------------\n')
            print(f"Number of insitu data points before outlier filtering: {insitu_band.count()}")
            print(f"Number of sat data points before outlier filtering: {sat_band.count()}")
            
            if time is not None:
                print(f"Number of time data points before outlier filtering: {time.count()}")
            else:
                print(f'Time parameter is not set.')

            result = remove_outliers(self, *args, **kwargs)
                
            if time is not None:
                insitu_band, sat_band, time = result
                print('\n----------------------------------------------------------------------------\n')
                print(f"Number of insitu data points after {int(threshold * 100)}% outlier filtering: {insitu_band.count()}")
                print(f"Number of sat data points after {int(threshold * 100)}% outlier filtering: {sat_band.count()}")
                print(f"Number of time data points after {int(threshold * 100)}% outlier filtering: {time.count()}")
            else:
                insitu_band, sat_band = result
                print('\n------------------------------------------------------------------\n')
                print(f"Number of insitu data points after {int(threshold * 100)}% outlier filtering: {insitu_band.count()}")
                print(f"Number of sat data points after {int(threshold * 100)}% outlier filtering: {sat_band.count()}")
            print(f'\n----------------------------------------------------------------------------')
            print(f'\tEnding of Remove outliers method debugger: Filtering threshold: {threshold * 100}%')
            print('----------------------------------------------------------------------------\n')
        return result
    return wrapper
