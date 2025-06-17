import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
import seaborn as sns

warnings.filterwarnings('ignore')


def convert_mjd_to_month(mjd_series):
    """Convert MJD to month numbers using astropy Time"""
    dates = Time(mjd_series, format='mjd').datetime
    return pd.Series(dates).dt.month


def create_all_visualizations():
    # Load the saved CSV files
    print("Loading saved data files...")
    jwst_df = pd.read_csv('data_jwst_2023.csv')
    hst_df = pd.read_csv('data_hst_2023.csv')

    print(f"Loaded {len(jwst_df)} JWST observations")
    print(f"Loaded {len(hst_df)} HST observations")

    # First set of visualizations (original 4 plots)
    plt.figure(figsize=(15, 10))

    # 1. Exposure Time Comparison
    plt.subplot(2, 2, 1)
    exposure_data = [
        jwst_df['t_exptime'].dropna(),
        hst_df['t_exptime'].dropna()
    ]
    plt.boxplot(exposure_data, labels=['JWST', 'HST'])
    plt.ylabel('Exposure Time (seconds)')
    plt.title('Exposure Time Comparison (2023)')

    # 2. Monthly Observation Counts
    plt.subplot(2, 2, 2)
    months = range(1, 13)

    # Convert MJD to months
    jwst_months = convert_mjd_to_month(jwst_df['t_min'])
    hst_months = convert_mjd_to_month(hst_df['t_min'])

    # Count observations per month
    jwst_monthly = jwst_months.value_counts().sort_index()
    hst_monthly = hst_months.value_counts().sort_index()

    # Ensure all months are represented
    all_months = pd.Series(index=months, data=0)
    jwst_monthly = jwst_monthly.combine_first(all_months)
    hst_monthly = hst_monthly.combine_first(all_months)

    plt.plot(months, jwst_monthly, label='JWST', marker='o')
    plt.plot(months, hst_monthly, label='HST', marker='o')
    plt.xlabel('Month')
    plt.ylabel('Number of Observations')
    plt.title('Monthly Observations (2023)')
    plt.legend()

    # 3. Top Targets
    plt.subplot(2, 2, 3)
    jwst_targets = jwst_df['target_name'].value_counts().head(10)
    plt.pie(jwst_targets.values, labels=jwst_targets.index, autopct='%1.1f%%')
    plt.title('Top 10 JWST Targets (2023)')

    # 4. Instrument Usage
    plt.subplot(2, 2, 4)
    instruments = jwst_df['instrument_name'].value_counts()
    plt.bar(range(len(instruments)), instruments.values)
    plt.xticks(range(len(instruments)), instruments.index, rotation=45)
    plt.title('JWST Instrument Usage (2023)')

    plt.tight_layout()
    plt.savefig('telescope_comparison_2023.png')
    plt.close()

    # Second set of visualizations (additional analysis - first 2 plots)
    plt.figure(figsize=(15, 6))

    # 5. Exposure Time Distribution (Density Plot)
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=jwst_df['t_exptime'].dropna(), label='JWST', alpha=0.5)
    sns.kdeplot(data=hst_df['t_exptime'].dropna(), label='HST', alpha=0.5)
    plt.xlabel('Exposure Time (seconds)')
    plt.ylabel('Density')
    plt.title('Exposure Time Distribution Comparison')
    plt.legend()

    # 6. Target Overlap Analysis
    plt.subplot(1, 2, 2)
    jwst_targets = jwst_df['target_name'].value_counts()
    hst_targets = hst_df['target_name'].value_counts()
    common_targets = set(jwst_targets.index).intersection(set(hst_targets.index))
    only_jwst = set(jwst_targets.index) - common_targets
    only_hst = set(hst_targets.index) - common_targets

    plt.pie([len(only_jwst), len(common_targets), len(only_hst)],
            labels=['JWST Only', 'Common', 'HST Only'],
            autopct='%1.1f%%',
            colors=['lightblue', 'green', 'orange'])
    plt.title('Target Distribution Between Telescopes')

    plt.tight_layout()
    plt.savefig('additional_analysis_2023.png')
    plt.close()

    # Third set of visualizations (additional analysis - last 2 plots)
    plt.figure(figsize=(15, 6))

    # 7. Observation Duration vs Time of Year
    plt.subplot(1, 2, 1)
    jwst_dates = Time(jwst_df['t_min'], format='mjd').datetime
    hst_dates = Time(hst_df['t_min'], format='mjd').datetime

    jwst_doy = pd.Series(jwst_dates).dt.dayofyear
    hst_doy = pd.Series(hst_dates).dt.dayofyear

    plt.scatter(jwst_doy, jwst_df['t_exptime'],
                alpha=0.3, label='JWST', s=5)
    plt.scatter(hst_doy, hst_df['t_exptime'],
                alpha=0.3, label='HST', s=5)
    plt.xlabel('Day of Year')
    plt.ylabel('Exposure Time (seconds)')
    plt.title('Observation Duration Throughout 2023')
    plt.legend()

    # 8. Cumulative Observation Count
    plt.subplot(1, 2, 2)
    jwst_sorted = pd.Series(jwst_dates).sort_values()
    hst_sorted = pd.Series(hst_dates).sort_values()

    jwst_cumcount = range(1, len(jwst_sorted) + 1)
    hst_cumcount = range(1, len(hst_sorted) + 1)

    plt.plot(jwst_sorted, jwst_cumcount, label='JWST')
    plt.plot(hst_sorted, hst_cumcount, label='HST')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Number of Observations')
    plt.title('Cumulative Observations in 2023')
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('additional_analysis_2023_part2.png')
    plt.close()

    # Save comprehensive analysis to text file
    with open('comprehensive_analysis_2023.txt', 'w') as f:
        f.write("Comprehensive Analysis of 2023 Observations\n")
        f.write("========================================\n\n")

        f.write("Basic Statistics:\n")
        f.write(f"Total JWST Observations: {len(jwst_df)}\n")
        f.write(f"Total HST Observations: {len(hst_df)}\n")
        f.write(f"JWST Unique Targets: {jwst_df['target_name'].nunique()}\n")
        f.write(f"HST Unique Targets: {hst_df['target_name'].nunique()}\n\n")

        f.write("Target Distribution:\n")
        f.write(f"JWST-only targets: {len(only_jwst)}\n")
        f.write(f"HST-only targets: {len(only_hst)}\n")
        f.write(f"Common targets: {len(common_targets)}\n\n")

        f.write("Exposure Time Statistics:\n")
        f.write(f"JWST median exposure: {jwst_df['t_exptime'].median():.2f} seconds\n")
        f.write(f"HST median exposure: {hst_df['t_exptime'].median():.2f} seconds\n")
        f.write(f"JWST max exposure: {jwst_df['t_exptime'].max():.2f} seconds\n")
        f.write(f"HST max exposure: {hst_df['t_exptime'].max():.2f} seconds\n")


if __name__ == "__main__":
    print("Creating comprehensive analysis...")
    create_all_visualizations()
    print("\nAnalysis completed! Generated files:")
    print("1. telescope_comparison_2023.png - Original 4 visualizations")
    print("2. additional_analysis_2023.png - 2 additional visualizations")
    print("3. additional_analysis_2023_part2.png - 2 more visualizations")
    print("4. comprehensive_analysis_2023.txt - Complete statistical analysis")