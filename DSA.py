from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# Descriptive statistical analysis

class DSA:
    def __init__(self, df):
        self.df = df
    
    def get_summary(self):
        return self.df.describe()
    

class DSA_CENTRAL_TENDENSIES(DSA):
    
    def __init__(self):
        super().__init__()
    
    def calculate_average(self, column_name):
        return self.df[column_name].mean()
    
    def calculate_median(self, column_name):
        return self.df[column_name].median()
    
    def calculate_mode(self, column_name):
        return self.df[column_name].mode().values[0]
    
    
class DSA_VARIABILITY(DSA):
    
    def __init__(self):
        super().__init__()
        
    def calculate_interquartile_range(self, column_name):
        return self.df[column_name].quantile(0.75) - self.df[column_name].quantile(0.25)
    
    def calculate_range(self, column_name):
        return self.df[column_name].max() - self.df[column_name].min()
    
    def standard_deviation(self, column_name):
        return self.df[column_name].std()
    
    def variance(self, column_name):
        return self.df[column_name].var()
    
    
class DSA_SHAPE(DSA):
    
    def __init__(self):
        super().__init__()
    
    def skewness(self, column_name):
        return self.df[column_name].skew()
    
    def kurtosis(self, column_name):
        return self.df[column_name].kurtosis()
    

class DSA_MEASURE_OF_POSITION(DSA):
    
    def __init__(self):
        super().__init__()
    
    def measure_quartiles(self, column_name):
        return self.df[column_name].quantile([0.25, 0.5, 0.75])
    
    def measure_percentiles(self, column_name):
        return self.df[column_name].quantile(list(range(101)))
    
    def calculate_z_scores(self, column_name):
        return (self.df[column_name] - self.df[column_name].mean()) / self.df[column_name].std()
    

class DSA_FREQUENCY_DISTIBUTION(DSA):
    
    def __init__(self):
        super().__init__()
    
    def show_frequency_table(self, column_name):
        return pd.DataFrame({
                'Frequency': self.df[column_name].value_counts(),
                'Relative Frequency': self.df[column_name].value_counts(normalize=True)
                })

    def plot_histogram(self, column_name):
        plt.hist(self.df[column_name])
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {column_name}')
        plt.show()  
    
    
    
class DSA_OTHERS(DSA):
    
    def __init__(self):
        super().__init__()
      
    def minimum(self, column_name):
        return self.df[column_name].min()
    
    def maximum(self, column_name):
        return self.df[column_name].max()
    
 
    def calculate_correlation(self, column1, column2):
        return self.df[column1].corr(self.df[column2])
    
        
    def plot_bar(self, column_name):
        self.df[column_name].value_counts().plot(kind='bar')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.title(f'Bar plot of {column_name}')
        plt.show()
        
        

class DSA_RELATIONSHIP(DSA):
      # ["artists","id","name","loudness","release_date"]
    def __init__(self):
        super().__init__()
    
    def calculate_correlation_between_features(self, type ,column1, column2):
        # return self.df.get(column1).value

        if type == 'pearson':
            # Pearson correlation
            pearson_corr = self.df[column1].corr(self.df[column2], method='pearson')
            return "Pearson Correlation:", pearson_corr

        elif type =='spearman':
            # Spearman correlation
            spearman_corr = self.df[column1].corr(self.df[column2], method='spearman')
            return "Spearman Correlation:", spearman_corr
        
        elif type == 'kendall':
            # Kendall Tau correlation
            kendall_corr = self.df[column1].corr(self.df[column2], method='kendall')
            return "Kendall Tau Correlation:", kendall_corr
        
        else:
            return "Invalid correlation type"
        
    def plot_corrolation_table(self, excluded_columns_name):
        data = self.df
        data.drop(columns= excluded_columns_name,inplace=True)
        corr = data.corr()
        plt.subplots(figsize = (10,8))
        sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, annot = True, cmap = sns.diverging_palette(105, 255, as_cmap = True))
      