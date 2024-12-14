import pandas as pd

def time_to_seconds(time_column):
    return (time_column.dt.hour * 3600 + time_column.dt.minute * 60 + time_column.dt.second).astype('int64')