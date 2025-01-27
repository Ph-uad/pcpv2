import os
import pandas as pd


class DATA_WRANGLER:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        # Load data from the specified file path and if file path does not exist raise FileNotFoundError
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        # else load the data from the specified file path
        return pd.read_csv(self.file_path)

    def detect_mixed_columns(self, df):
        mixed_columns = []
        # Check for mixed type columns in the data and store them in the mixed_columns list
        for col in df.columns:
            types = df[col].map(type).unique()
            # Append the column name to mixed_columns if more than one type is detected
            if len(types) > 1:
                mixed_columns.append((col, types))
        return mixed_columns

    def coerce_mixed_columns(self, df):
        mixed_columns = []
        # Coerce the mixed type columns to numeric and drop the rows with missing values
        for column in self.detect_mixed_columns(df):
            mixed_columns.append(column[0])
        for column in mixed_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        df.dropna(inplace=True)
        return df

    def load_and_clean_data(self):
        df = self.load_data()
        return self.coerce_mixed_columns(df)
