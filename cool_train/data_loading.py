import pandas as pd


type_dict = {'vehicle_id': 'int32',
             'timestamps_UTC': 'datetime64[ns]', 
             'lat':'float64',
             'lon': 'float64',
             'RS_E_InAirTemp_PC1':'float64',
             'RS_E_InAirTemp_PC2':'float64', 
             'RS_E_OilPress_PC1':'float64',
             'RS_E_OilPress_PC2':'float64', 
             'RS_E_RPM_PC1':'float64',
             'RS_E_RPM_PC2':'float64', 
             'RS_E_WatTemp_PC1':'float64',
             'RS_E_WatTemp_PC2':'float64', 
             'RS_T_OilTemp_PC1':'float64', 
             'RS_T_OilTemp_PC2':'float64'}


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

    for column, desired_type in column_types.items():
        if column in corrected_dataframe.columns:
            # Removing non-numeric columns
            if desired_type != 'datetime64[ns]':
                # Transform to string, then replace the comas with point
                corrected_dataframe[column] = corrected_dataframe[column].astype(str).str.replace(',', '.', regex=False)
                # Change all values to numeric
                corrected_dataframe[column] = pd.to_numeric(corrected_dataframe[column])   
                
            # Format right dtype                
            corrected_dataframe[column] = corrected_dataframe[column].astype(desired_type)
                
    return corrected_dataframe

