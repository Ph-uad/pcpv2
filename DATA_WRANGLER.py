import os
import pandas as pd


class DATA_WRANGLER:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load_data(self):
        # Load data from the specified file path and if file path does not exist raise FileNotFoundError
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        #else load the data from the specified file path
        return pd.read_csv(self.file_path)
    
    def detect_mixed_columns(self):
        mixed_columns = []
        # Check for mixed type columns in the data and store them in the mixed_columns list
        for col in self.data.columns:
            types = self.data[col].map(type).unique()
            #append types to the mixed_columns list if the length of types is greater than 1
            for type in types:
                mixed_columns.append((col, types))
        return mixed_columns
    
    def coerce_mixed_columns(self):
        mixed_columns = []
        # Coerce the mixed type columns to numeric and drop the rows with missing values
        for column in self.detect_mixed_columns():
            mixed_columns.append(column[0])
        for column in mixed_columns:
            self.data[column] = pd.to_numeric(self.data[column], errors='coerce')
        self.data.dropna(inplace=True)
        return self.data
    
    def load_and_clean_data(self):
        self.data = self.load_data()
        return self.coerce_mixed_columns()