import pandas as pd
import pytz
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from math import radians, sin, cos, sqrt, atan2


# Time-related features
def add_time_features(df, timestamp_column_UTC='timestamps_UTC'):
    """
    Add time-related features to a DataFrame.

    Parameters:
    - df: DataFrame, input DataFrame with timestamp_column
    - timestamp_column: str, name of the timestamp column

    Returns:
    - DataFrame, DataFrame with added time-related features
    """
    # Add extra column with local time (UTC+2)
    brussels_tz = pytz.timezone('Europe/Brussels')
    df[timestamp_column_UTC] = pd.to_datetime(df[timestamp_column_UTC], utc=True)
    df['timestamps_local'] = df[timestamp_column_UTC].dt.tz_convert(brussels_tz)
    
    timestamp_column = 'timestamps_local'

    # Extract time-related features
    df['month'] = df[timestamp_column].dt.month
    df['day'] = df[timestamp_column].dt.day
    df['hour'] = df[timestamp_column].dt.hour
    df['minute'] = df[timestamp_column].dt.minute
    df['second'] = df[timestamp_column].dt.second
    df['dayOfWeek'] = df[timestamp_column].dt.dayofweek
    df['isWeekend'] = df[timestamp_column].dt.weekday >= 5  # 5 and 6 correspond to Saturday and Sunday
    df['quarter'] = df[timestamp_column].dt.quarter
    df['timeOfDay'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])

    return df


# Space-related features

def haversine_distance(coord1, coord2):
    """
    Calculate the Haversine distance between two GPS coordinates.

    Parameters:
    - coord1: Tuple, (latitude, longitude) of the first point
    - coord2: Tuple, (latitude, longitude) of the second point

    Returns:
    - float, Haversine distance in kilometers
    """
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)

    # Calculate the change in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula to calculate distance
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance

# Function to calculate haversine distance between two rows of a DataFrame
def calculate_distance(row1, row2):
    # Extract latitude and longitude from the rows
    coord1 = (row1['lat'], row1['lon'])
    coord2 = (row2['lat'], row2['lon'])

    # Calculate haversine distance
    distance = haversine_distance(coord1, coord2)
    return distance


def compute_spatial_features(df, lat_col='lat', lon_col='lon', time_col='timestamps_UTC'):
    """
    Compute spatial features for GPS data.

    Parameters:
    - df: DataFrame, input DataFrame with GPS data
    - lat_col: str, name of the latitude column
    - lon_col: str, name of the longitude column
    - time_col: str, name of the timestamp column

    Returns:
    - DataFrame, DataFrame with computed spatial features
    """
    # Sort the DataFrame by timestamp
    df = df.sort_values(by=time_col)

    # Compute time differences
    df['TimeDiff'] = df[time_col].diff().dt.total_seconds()

    # Compute distance and speed
    print('Computing Distance...')
    df['Distance'] = df.shift().apply(lambda row: calculate_distance(row, df.loc[row.name]), axis=1)

    df['Speed'] = df['Distance'] / df['TimeDiff']

    # Compute heading or bearing
    df['Heading'] = np.arctan2(np.sin(np.radians(df[lon_col] - df[lon_col].shift(1))) * np.cos(np.radians(df[lat_col])),
                               np.cos(np.radians(df[lat_col].shift(1))) * np.sin(np.radians(df[lat_col])) -
                               np.sin(np.radians(df[lat_col].shift(1))) * np.cos(np.radians(df[lat_col])) *
                               np.cos(np.radians(df[lon_col] - df[lon_col].shift(1))))
    df['Heading'] = np.degrees(df['Heading'])

    # Compute acceleration
    df['Acceleration'] = df['Speed'].diff() / df['TimeDiff']

    # Compute spatial bounds (Bounding Box)
    min_lat, max_lat = df[lat_col].min(), df[lat_col].max()
    min_lon, max_lon = df[lon_col].min(), df[lon_col].max()

    # Drop intermediate columns
    df = df.drop(['TimeDiff'], axis=1)

    return df