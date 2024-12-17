# from dataclasses import dataclass
from typing import *
import pandas as pd
from pathlib import Path
from enum import Enum

from rrs_data_debuger import *

# dashes for separating the output
dashes = "-" * 40

class RRSBandName(Enum):
    INSITU_BAND_412 = 'rrs_412_insitu'
    INSITU_BAND_443 = 'rrs_443_insitu'
    INSITU_BAND_490 = 'rrs_490_insitu'
    INSITU_BAND_510 = 'rrs_510_insitu'
    INSITU_BAND_560 = 'rrs_560_insitu'
    INSITU_BAND_665 = 'rrs_665_insitu'
    SAT_BAND_412 = 'rrs_412_sat'
    SAT_BAND_443 = 'rrs_443_sat'
    SAT_BAND_490 = 'rrs_490_sat'
    SAT_BAND_510 = 'rrs_510_sat'
    SAT_BAND_560 = 'rrs_560_sat'
    SAT_BAND_665 = 'rrs_665_sat'

class RRSVariables(Enum):
    IDX = 'idx'
    TIME = 'time'
    LAT = 'lat'
    LON = 'lon'
    ETOPO1 = 'etopo1'

class RRSData:
    def __init__(self, filename: str) -> None:
        """
        Initializes the RRSData object with the data from the specified file.

        :param filename: The name of the file to read the data from.
        :type filename: str
        :raises ValueError: If the file name is not a string.
        :raises ValueError: If the file type is not '.csv' or '.pkl'.
        :raises IOError: If there is an error reading the file.
        """
        # Check if the file name is a string
        if not isinstance(filename, str):
            raise ValueError('File name is not string type.')
        
        # Get the path to the file
        data_path: Path = Path(__file__).parent / filename

        # check if the file exists
        if not data_path.exists():
            raise FileNotFoundError(f'File not found: {data_path}')
        
        # Check if the file type is '.csv' or '.pkl'
        if data_path.suffix not in ['.csv', '.pkl']:
            raise ValueError(f'File type is not supported: {data_path.suffix}')
        try:
            # Read the file based on the file type
            if data_path.suffix == '.csv':
                print(f'Reading file: {filename}')
                print('DataFrame is created successfully')
                self.df: pd.DataFrame = pd.read_csv(data_path)
            elif data_path.suffix == '.pkl':
                print(f'Reading file: {filename}')
                print('DataFrame is created successfully\n')
                self.df: pd.DataFrame = pd.read_pickle(data_path)
        except (IOError, FileNotFoundError, pd.errors.EmptyDataError) as e:
            # Raise an IOError if there is an error reading the file
            raise IOError(f'Error reading file: {e}') from e

    def print_data_summary(self, select: Literal['info', 'summary', 'both'] = 'info') -> None:
        """
        Prints a summary of the data.

        Args:
            select (Literal['info', 'summary', 'both']): The type of summary to print. Defaults to 'both'.

        Returns:
            None
        """
        if select == 'info':
            print('Data Info:')
            print(self.df.info())
        elif select == 'summary':
            print('Data Summary:')
            print(self.df.describe())
        elif select == 'both':
            print('Data Head:')
            print(self.df.info())
            print('Data Summary:')
            print(self.df.describe())

    @property
    def idx(self) -> int:
        return self.df[RRSVariables.IDX.value]
    
    @property
    def time(self) -> pd.Series:
        return self.df[RRSVariables.TIME.value]
    
    @property
    def coordinates(self) -> Tuple[pd.Series, pd.Series]:
        return self.df[RRSVariables.LAT.value], self.df[RRSVariables.LON.value]
    
    @property
    def etopo1(self) -> pd.Series:
        return self.df[RRSVariables.ETOPO1.value]
    
    @property
    def rrs_insitu_bands(self) -> pd.DataFrame:
        return self.df.loc[:, f'{RRSBandName.INSITU_BAND_412.value}': f'{RRSBandName.INSITU_BAND_665.value}']
    
    @property
    def rrs_sat_bands(self) -> pd.DataFrame:
        return self.df.loc[:, f'{RRSBandName.SAT_BAND_412.value}': f'{RRSBandName.SAT_BAND_665.value}']
    
    def get_rrs_band(self, band: 'RRSBandName') -> pd.Series:
        return self.df[f'{band.value}']
    
    def get_rrs_pair(self, insitu_band: 'RRSBandName', sat_band: 'RRSBandName') -> Tuple[pd.Series, pd.Series]:
        return self.get_rrs_band(insitu_band), self.get_rrs_band(sat_band)

def main():
    # filename = 'insitudb_rrs_satbands6_final_total_cleaned_20240705_225251-tr0.0-ML-ready-20240810.pkl'
    filename = "insitudb_rrs_satbands6_final_total_cleaned_20240705_225251.pkl"
    rrs_data = RRSData(filename)
    etopo1 = RRSVariables.ETOPO1
    print(etopo1, type(etopo1))
    print(etopo1.name, type(etopo1.name))
    print(etopo1.value, type(etopo1.value))

    time = rrs_data.time
    # print(time.info())
    # idx = rrs_data.idx
    # print(idx)
    print('\n----------------------------------------------\n')
    # rrs_412_insitu, rrs_412_sat, _ = rrs_data.mask_nan_values(412,412,time=time, debug=True)
    # rrs_412_insitu, rrs_412_sat, rrs_412_time = rrs_data.mask_nan_values(RRSBandName.INSITU_BAND_412, RRSBandName.SAT_BAND_412 ,time=time, debug=True)

    # print('\n----------------------------------------------\n')

    # rrs_412_insitu_outp50, rrs_412_sat_outp50, rrs_412_time_outp50 = rrs_data.remove_outliers(rrs_412_insitu, rrs_412_sat, time=rrs_412_time, threshold=0.5, debug=True)


    # # print('\n----------------------------------------------\n')
    # rrs_443_insitu, rrs_443_sat, rrs_443_time = rrs_data.mask_nan_values(RRSBandName.INSITU_BAND_443, RRSBandName.SAT_BAND_443 ,time=time, debug=True)
    # # print('\n----------------------------------------------\n')
    # rrs_443_insitu_outp50, rrs_443_sat_outp50, rrs_443_time_outp50 = rrs_data.remove_outliers(rrs_443_insitu, rrs_443_sat, time=rrs_443_time, threshold=0.5, debug=True)

    # print('\n---------------------------------------------------\n')
    # print(rrs_443_insitu.info())
    # print('\n----------------------------------------------\n')
    # print(rrs_443_sat.info())
    # print('\n----------------------------------------------\n')
    # print(rrs_443_time.info())
    # print('\n----------------------------------------------\n')
    
    # rrs_443_insitu_outp50, rrs_443_sat_outp50 = rrs_data.remove_outliers(rrs_443_insitu, rrs_443_sat, threshold=0.5, time=None, debug=True)
    # print(rrs_443_insitu_outp50)
    # print(rrs_443_sat_outp50)

    # print(rrs_443_insitu_outp50.info())
    # print(rrs_443_sat_outp50.info())

    rrs_443_insitu = rrs_data.get_rrs_band(RRSBandName.INSITU_BAND_443)
    print(rrs_443_insitu.info())


if __name__ == "__main__":
    main()