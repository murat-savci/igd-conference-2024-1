import matplotlib.pyplot as plt

from rrs_data import RRSData, RRSBandName, RRSVariables
from outlier_detector import OutlierDetector
from rrs_plotter import rrs_scatter_plot, rrs_histogram_plot, rrs_time_series_plot


def main():

    filename = "insitudb_rrs_satbands6_final_total_cleaned_20240705_225251.pkl"
    rrs_data: RRSData = RRSData(filename)

    # Check the filename to determine the spatial resolution
    if '1KM' in filename:
        spatial_resolution = 1
    else:
        spatial_resolution = 4
    


if __name__ == '__main__':
    main()