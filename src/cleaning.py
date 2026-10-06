import pandas as pd
import numpy as np

def clean_plot_data(df):
    """Clean plots data."""
    df = df.copy()
    df.replace([-999, '-999', -999.0], np.nan, inplace=True)
    if 'region' in df.columns:
        df['region'] = df['region'].str.lower().str.strip()
    if 'crop_type' in df.columns:
        df['crop_type'] = df['crop_type'].str.lower().str.strip()
    return df

def clean_weather_data(df):
    """Clean weather data."""
    df = df.copy()
    df.replace([-999, '-999', -999.0], np.nan, inplace=True)
    if 'region' in df.columns:
        df['region'] = df['region'].str.lower().str.strip()
    return df

def clean_prices_data(df):
    """Clean market prices data."""
    df = df.copy()
    df.replace([-999, '-999', -999.0], np.nan, inplace=True)
    if 'region' in df.columns:
        df['region'] = df['region'].str.lower().str.strip()
    if 'crop_type' in df.columns:
        df['crop_type'] = df['crop_type'].str.lower().str.strip()
    return df
