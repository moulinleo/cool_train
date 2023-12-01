import pandas as pd

# Filtre 1: Impossible values
# -> Sensor problem
param_impossible_values = {'lon': {'min': 0, 'max': 10}, # 
              'lat': {'min': 20, 'max': 60},
              'RS_E_InAirTemp_PC1': {'min': 0, 'max': 65},
              'RS_E_InAirTemp_PC2': {'min': 0, 'max': 65},
              'RS_E_OilPress_PC1': {'min': 0, 'max': 1000},
              'RS_E_OilPress_PC2': {'min': 0, 'max': 1000},
              'RS_E_RPM_PC1': {'min': 0, 'max': 2500},
              'RS_E_RPM_PC2': {'min': 0, 'max': 2500},
              'RS_E_WatTemp_PC1': {'min': 0, 'max': 100},
              'RS_E_WatTemp_PC2': {'min': 0, 'max': 100},
              'RS_T_OilTemp_PC1': {'min': 0, 'max': 115},
              'RS_T_OilTemp_PC2': {'min': 0, 'max': 115}
}

def filter_impossible_values(df, filter_params=param_impossible_values):
    """
    Filter a DataFrame based on given conditions for each column.

    Parameters:
    - df: pandas DataFrame
    - filter_params: dictionary containing filtering conditions for each column

    Returns:
    - filtered DataFrame
    """
    # Create an initial filter with all True values
    initial_filter = pd.Series([True] * len(df), index=df.index)

    # Apply filters for each column based on the provided conditions
    for column, conditions in filter_params.items():
        min_condition = df[column] >= conditions.get('min', float('-inf'))
        max_condition = df[column] <= conditions.get('max', float('inf'))
        initial_filter = initial_filter & min_condition & max_condition

    # Apply the final filter to the DataFrame
    filtered_df = df[initial_filter]

    return filtered_df


# Filter 2: Cases where both PC are off (RPM=0)

def filter_zeros(df, zero_columns=['RS_E_RPM_PC1', 'RS_E_RPM_PC2']):
    """
    Filter cases where variables specified in zero_columns are equal to 0 simultaneously.

    Parameters:
    - df: pandas DataFrame
    - zero_columns: list of column names to check for zero values

    Returns:
    - filtered DataFrame
    """
    non_zero_filter = pd.Series([True] * len(df), index=df.index)

    for column in zero_columns:
        non_zero_condition = df[column] != 0
        non_zero_filter = non_zero_filter & non_zero_condition

    non_zero_filtered_df = df[non_zero_filter]

    return non_zero_filtered_df