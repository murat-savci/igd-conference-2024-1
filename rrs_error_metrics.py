import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


from typing import Dict

def rrs_calculate_error_metrics(
    insitu_band: pd.Series, sat_band: pd.Series
) -> Dict[str, str]:
    """
    Calculate various error metrics between two pandas Series objects.

    Parameters:
        insitu_band (pd.Series): The in-situ band values.
        sat_band (pd.Series): The satellite band values.

    Returns:
        dict: A dictionary containing the calculated error metrics. The keys are:
            - 'Rrs Band': The RRS band.
            - 'N': Number of data points.
            - 'MAE': Mean Absolute Error.
            - 'RMSE': Root Mean Squared Error.
            - 'R2': R-squared.
            - 'Mean bias': Mean Bias Error.
            - 'CV': Coefficient of Variation.
            - 'Slope': Slope of the regression line.
            - 'Intercept': Intercept of the regression line.
    """
    insitu_band_name = insitu_band.name
    rrs_band = insitu_band_name.replace('_insitu', '').replace('_', ' ').capitalize()
    total_points = len(insitu_band)
    r2 = r2_score(insitu_band, sat_band)
    mae = mean_absolute_error(insitu_band, sat_band)
    rmse = np.sqrt(mean_squared_error(insitu_band, sat_band))
    bias = np.mean(insitu_band - sat_band) # Mean bias error
    coefficient_of_variation = rmse / np.mean(insitu_band) # Coefficient of Variation
    slope, intercept = np.polyfit(insitu_band, sat_band, 1)
    return {
        'Band': rrs_band,
        'N': str(total_points),
        'MAE': f'{mae:.4f}',
        'RMSE': f'{rmse:.4f}',
        'R2': f'{r2:.2f}',
        'Mean bias': f'{bias:.4f}',
        'CV': f'{coefficient_of_variation:.2f}',
        'Slope': f'{slope:.2f}',
        'Intercept': f'{intercept:.3f}'
    }

def print_error_metrics_to_dataframe(
    rrs_error_metrics: Dict[str, str]
) -> pd.DataFrame:
    """
    Print the error metrics to a pandas DataFrame.

    Parameters:
        rrs_error_metrics (Dict[str, str]): The error metrics.

    Returns:
        pd.DataFrame: The error metrics in a DataFrame.
    """
    return pd.DataFrame([rrs_error_metrics])
