import pandas as pd


type_dict = {'vehicle_id': 'int32', 
             'timestamps_UTC': 'datetime64[ns]'}


def correct_column_types(dataframe, column_types=type_dict):
    """
    Corrects the type of each column in a DataFrame based on the specified types in the dictionary.

    Parameters:
    - dataframe: pandas DataFrame
    - column_types: dictionary where keys are column names and values are the desired types

    Returns:
    - corrected_dataframe: pandas DataFrame with corrected column types
    """
    corrected_dataframe = dataframe.copy()

    for column in corrected_dataframe.columns:
        # Default type is float64
        desired_type = 'float64'
        
        # Check if a different type is specified in the dictionary
        if column in column_types:
            desired_type = column_types[column]

        if desired_type == 'datetime64[ns]':
            # Convert to datetime
            corrected_dataframe[column] = pd.to_datetime(corrected_dataframe[column], errors='coerce')
        
        else:
            # Convert to string, replace commas with points, and convert to numeric
            corrected_dataframe[column] = pd.to_numeric(corrected_dataframe[column].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        
        # Convert to the desired type
        corrected_dataframe[column] = corrected_dataframe[column].astype(desired_type)
    
    return corrected_dataframe


