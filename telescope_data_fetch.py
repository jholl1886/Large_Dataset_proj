import warnings
from astroquery.mast import Observations
import pandas as pd
from astropy.time import Time

warnings.filterwarnings('ignore')


def get_2023_observations(telescope):
    """Get only 2023 observations for specified telescope"""
    print(f"\nQuerying {telescope} observations for 2023...")

    # Convert dates to MJD (Modified Julian Date)
    start_date = Time("2023-01-01 00:00:00").mjd
    end_date = Time("2023-12-31 23:59:59").mjd

    query_params = {
        'project': telescope,
        'obs_collection': telescope,
        'dataRights': 'PUBLIC',
        't_min': [start_date, end_date]
    }

    try:
        obs_table = Observations.query_criteria(**query_params)
        if len(obs_table) > 0:
            df = obs_table.to_pandas()
            # Save the data
            filename = f"data_{telescope.lower()}_2023.csv"
            df.to_csv(filename)
            print(f"Found {len(df)} observations for {telescope}")
            print(f"Saved data to {filename}")
            return df
        else:
            print(f"No observations found for {telescope} in 2023")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error querying {telescope}: {str(e)}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("Starting data collection for 2023...")

    # Get JWST data
    jwst_df = get_2023_observations('JWST')

    # Get HST data
    hst_df = get_2023_observations('HST')

    print("\nData collection completed!")
    print("Generated files:")
    print("1. data_jwst_2023.csv - JWST observations")
    print("2. data_hst_2023.csv - HST observations")