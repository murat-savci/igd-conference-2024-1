import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

def read_data(filename: str) -> pd.DataFrame:
    """
    Reads a csv or pickle file and returns a pandas DataFrame.

    Args:
        filename (str): The name of the file to be read.

    Returns:
        pd.DataFrame: A pandas DataFrame if the file is successfully read.

    Raises:
        ValueError: If the file name is not a string.
        ValueError: If the file type is not '.csv' or '.pkl'.
        IOError: If there is an error reading the file.
    """

    # Check if the file name is a string
    if not isinstance(filename, str):
        raise ValueError('File name is not string type.')
    
    # Get the path to the file
    data_path = Path(__file__).parent / filename
    
    # Check if the file type is '.csv' or '.pkl'
    if data_path.suffix not in ['.csv', '.pkl']:
        raise ValueError(f'File type is not supported: {data_path.suffix}')
    
    try:
        # Read the file based on the file type
        if data_path.suffix == '.csv':
            print(f'Reading file: {filename}')
            return pd.read_csv(data_path)
        elif data_path.suffix == '.pkl':
            print(f'Reading file: {filename}')
            return pd.read_pickle(data_path)
    except (IOError, FileNotFoundError, pd.errors.EmptyDataError) as e:
        # Raise an IOError if there is an error reading the file
        raise IOError(f'Error reading file: {e}') from e

def mask_nan_outliers(
    insitu_band_val: pd.Series, 
    sat_band_val: pd.Series, 
    threshold: float = 0, 
    time: Optional[pd.Series] = None, 
    debug: bool = False
) -> Tuple[pd.Series, pd.Series, Optional[pd.Series]]:
    """
    Masks out NaN values and outliers from the in-situ and satellite band values.

    Parameters:
        insitu_band_val (pd.Series): The in-situ band values.
        sat_band_val (pd.Series): The satellite band values.
        threshold (float, optional): The percentage threshold for outliers. Defaults to 0.
        time (Optional[pd.Series], optional): The time data. Defaults to None.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        Tuple[pd.Series, pd.Series, Optional[pd.Series]]: A tuple containing the filtered in-situ band values, 
        the filtered satellite band values, and the filtered time data (if provided).
    """

    # dashes for separating the output
    dashes = "-" * 40

    if debug:
         # remove only _insitu from the band name
        band_name = insitu_band_val.name.strip('_insitu')
        print(f"\n{dashes}\t{band_name}\t{dashes}\n")
        print(f"Number of insitu data points before filtering: {insitu_band_val.count()}")
        print(f"Number of sat data points before filtering: {sat_band_val.count()}")
   
    # Filter out NaN values
    mask = ~np.isnan(insitu_band_val) & ~np.isnan(sat_band_val)
    insitu_band_val = insitu_band_val[mask]
    sat_band_val = sat_band_val[mask]
    if time is not None:
        time = time[mask]

    if debug:
        print(f"Number of insitu data points after NAN filtering: {insitu_band_val.count()}")
        print(f"Number of sat data points after NAN filtering: {sat_band_val.count()}")

    if not threshold == 0: # threshold is zero means no outlier removal
        # If satellite band is percentage% or more bigger or smaller than in-situ band, then remove both values
        mask = (sat_band_val > insitu_band_val * (1 + threshold)) | (sat_band_val < insitu_band_val * (1 - threshold))
        insitu_band_val = insitu_band_val[~mask]
        sat_band_val = sat_band_val[~mask]
        if time is not None:
            time = time[~mask]

    if debug:
        no_of_filtered_data_points = insitu_band_val.count() # Number of data points after filtering
        print(f"Number of data points after {threshold * 100}% filtering: {no_of_filtered_data_points}")

    return insitu_band_val, sat_band_val, time

def mask_all_nan_rows(df:pd.DataFrame, debug=False) -> pd.DataFrame:
    """
    Masks all rows in a DataFrame if they all contain NaN values in the specified columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to be processed.
        
    Returns:
        pd.DataFrame: The DataFrame with rows containing NaN values in the specified columns removed.
    """
    mask = (df['rrs_412_insitu'].notna() & df['rrs_412_sat'].notna()) | (df['rrs_443_insitu'].notna() & df['rrs_443_sat'].notna()) | (df['rrs_490_insitu'].notna() & df['rrs_490_sat'].notna()) | (df['rrs_510_insitu'].notna() & df['rrs_510_sat'].notna()) | (df['rrs_560_insitu'].notna() & df['rrs_560_sat'].notna()) | (df['rrs_665_insitu'].notna() & df['rrs_665_sat'].notna())
    df = df[mask]

    if debug:
        print(f"Number of data points after removing rows with NaN values: {df.shape[0]}")
        print(mask)
        print(mask.value_counts())

    return df

def detect_outliers_difference_iqr(
    insitu_band_val: pd.Series, 
    sat_band_val: pd.Series, 
    threshold: float = 1.5, 
    debug: bool = False
) -> Tuple[pd.Series, pd.Series]:
    """
    Detects outliers in the difference between in-situ and satellite band values using the Interquartile Range (IQR) method.

    Parameters:
        insitu_band_val (pd.Series): The in-situ band values.
        sat_band_val (pd.Series): The satellite band values.
        threshold (float, optional): The threshold for outlier detection. Defaults to 0.5.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        Tuple[pd.Series, pd.Series]: A tuple containing the filtered in-situ band values and the filtered satellite band values.
    """
    diff = insitu_band_val - sat_band_val
    Q1 = diff.quantile(0.25)
    Q3 = diff.quantile(0.75)
    IQR = Q3 - Q1

    if debug:
        print(f"Q1: {Q1}")
        print(f"Q3: {Q3}")
        print(f"IQR: {IQR}")

    mask = (diff < (Q1 - threshold * IQR)) | (diff > (Q3 + threshold * IQR))
    insitu_band_val = insitu_band_val[~mask]
    sat_band_val = sat_band_val[~mask]

    if debug:
        no_of_filtered_data_points = insitu_band_val.count() # Number of data points after filtering
        print(f"Number of data points after removing outliers: {no_of_filtered_data_points}")

    return insitu_band_val, sat_band_val

def detect_outliers_difference_std( insitu_band_val: pd.Series, sat_band_val: pd.Series, threshold: float = 3, debug: bool = False) -> Tuple[pd.Series, pd.Series]:
    """
    Detects outliers in the difference between in-situ and satellite band values using the standard deviation method.

    Parameters:
        insitu_band_val (pd.Series): The in-situ band values.
        sat_band_val (pd.Series): The satellite band values.
        threshold (float, optional): The threshold for outlier detection. Defaults to 3.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        Tuple[pd.Series, pd.Series]: A tuple containing the filtered in-situ band values and the filtered satellite band values.
    """
    diff = insitu_band_val - sat_band_val
    mean = diff.mean()
    std = diff.std()

    if debug:
        print(f'\ninsitu_band_val: {insitu_band_val.name}')
        print(f"Number of data points before filtering: {insitu_band_val.count()}")
        print(f"Mean: {mean}")
        print(f"Standard Deviation: {std}")

    mask = (diff < (mean - threshold * std)) | (diff > (mean + threshold * std))
    insitu_band_val = insitu_band_val[~mask]
    sat_band_val = sat_band_val[~mask]

    if debug:
        no_of_filtered_data_points = insitu_band_val.count() # Number of data points after filtering
        print(f"Number of data points after removing outliers: {no_of_filtered_data_points}")

    return insitu_band_val, sat_band_val


def main():

    from data_analysis_functions import my_r2_score
    from sklearn.metrics import r2_score



    filename = 'insitudb_rrs_satbands6_final_total_cleaned_20240705_225251-tr0.0-ML-ready-20240810.pkl'

    df = read_data(filename)


    reflectance_pairs = [
        ('rrs_412_insitu', 'rrs_412_sat'),
        ('rrs_443_insitu', 'rrs_443_sat'),
        ('rrs_490_insitu', 'rrs_490_sat'),
        ('rrs_510_insitu', 'rrs_510_sat'),
        ('rrs_560_insitu', 'rrs_560_sat'),
        ('rrs_665_insitu', 'rrs_665_sat')
    ]

    

    rrs_412_insitu, rrs_412_sat, _ = mask_nan_outliers(df[reflectance_pairs[0][0]], df[reflectance_pairs[0][1]], threshold=0., debug=True)
    rrs_443_insitu, rrs_443_sat, _ = mask_nan_outliers(df[reflectance_pairs[1][0]], df[reflectance_pairs[1][1]], threshold=0., debug=False)
    rrs_490_insitu, rrs_490_sat, _ = mask_nan_outliers(df[reflectance_pairs[2][0]], df[reflectance_pairs[2][1]], threshold=0., debug=False)
    rrs_510_insitu, rrs_510_sat, _ = mask_nan_outliers(df[reflectance_pairs[3][0]], df[reflectance_pairs[3][1]], threshold=0., debug=False)
    rrs_560_insitu, rrs_560_sat, _ = mask_nan_outliers(df[reflectance_pairs[4][0]], df[reflectance_pairs[4][1]], threshold=0., debug=False)
    rrs_665_insitu, rrs_665_sat, _ = mask_nan_outliers(df[reflectance_pairs[5][0]], df[reflectance_pairs[5][1]], threshold=0., debug=False)

    # threshold = 2
    threshold = 1.3
    rrs_412_insitu, rrs_412_sat = detect_outliers_difference_std(df[reflectance_pairs[0][0]], df[reflectance_pairs[0][1]], threshold=threshold, debug=True)
    rrs_443_insitu, rrs_443_sat = detect_outliers_difference_std(df[reflectance_pairs[1][0]], df[reflectance_pairs[1][1]], threshold=threshold, debug=True)
    rrs_490_insitu, rrs_490_sat = detect_outliers_difference_std(df[reflectance_pairs[2][0]], df[reflectance_pairs[2][1]], threshold=threshold, debug=True)
    rrs_510_insitu, rrs_510_sat = detect_outliers_difference_std(df[reflectance_pairs[3][0]], df[reflectance_pairs[3][1]], threshold=threshold, debug=True)
    rrs_560_insitu, rrs_560_sat = detect_outliers_difference_std(df[reflectance_pairs[4][0]], df[reflectance_pairs[4][1]], threshold=threshold, debug=True)
    rrs_665_insitu, rrs_665_sat = detect_outliers_difference_std(df[reflectance_pairs[5][0]], df[reflectance_pairs[5][1]], threshold=threshold, debug=True)

    
    # rrs_412_insitu, rrs_412_sat = detect_outliers_difference_iqr(df[reflectance_pairs[0][0]], df[reflectance_pairs[0][1]], threshold=1.5, debug=False)
    # rrs_443_insitu, rrs_443_sat = detect_outliers_difference_iqr(df[reflectance_pairs[1][0]], df[reflectance_pairs[1][1]], threshold=0.5, debug=False)
    # rrs_490_insitu, rrs_490_sat = detect_outliers_difference_iqr(df[reflectance_pairs[2][0]], df[reflectance_pairs[2][1]], threshold=0.5, debug=False)
    # rrs_510_insitu, rrs_510_sat = detect_outliers_difference_iqr(df[reflectance_pairs[3][0]], df[reflectance_pairs[3][1]], threshold=0.5, debug=False)
    # rrs_560_insitu, rrs_560_sat = detect_outliers_difference_iqr(df[reflectance_pairs[4][0]], df[reflectance_pairs[4][1]], threshold=0.5, debug=False)
    # rrs_665_insitu, rrs_665_sat = detect_outliers_difference_iqr(df[reflectance_pairs[5][0]], df[reflectance_pairs[5][1]], threshold=0.5, debug=False)

    print(f'rrs_412_insitu: {rrs_412_insitu.count()}')
    print(f'rrs_412_sat: {rrs_412_sat.count()}\n')
    

 
    r2_412 = my_r2_score(rrs_412_insitu, rrs_412_sat, debug=False)
    r2_443 = my_r2_score(rrs_443_insitu, rrs_443_sat, debug=False)
    r2_490 = my_r2_score(rrs_490_insitu, rrs_490_sat, debug=False)
    r2_510 = my_r2_score(rrs_510_insitu, rrs_510_sat, debug=False)
    r2_560 = my_r2_score(rrs_560_insitu, rrs_560_sat, debug=False)
    r2_665 = my_r2_score(rrs_665_insitu, rrs_665_sat, debug=False)

    print(f' r2_412 = {r2_412:.2f}\n')

    print(f' r2_412 = {r2_412:.2f}\n r2_443 = {r2_443:.2f}\n r2_490 = {r2_490:.2f}\n r2_510 = {r2_510:.2f}\n r2_560 = {r2_560:.2f}\n r2_665 = {r2_665:.2f}\n')



    # r2_412_sk = r2_score(rrs_412_insitu, rrs_412_sat)
    # r2_443_sk = r2_score(rrs_443_insitu, rrs_443_sat)
    # r2_490_sk = r2_score(rrs_490_insitu, rrs_490_sat)
    # r2_510_sk = r2_score(rrs_510_insitu, rrs_510_sat)
    # r2_560_sk = r2_score(rrs_560_insitu, rrs_560_sat)
    # r2_665_sk = r2_score(rrs_665_insitu, rrs_665_sat)

    # print(f'r2_412_sk = {r2_412_sk}\n, r2_443_sk = {r2_443_sk}\n, r2_490_sk = {r2_490_sk}\n, r2_510_sk = {r2_510_sk}\n, r2_560_sk = {r2_560_sk}\n, r2_665_sk = {r2_665_sk}\n')

if __name__ == "__main__":
    main()