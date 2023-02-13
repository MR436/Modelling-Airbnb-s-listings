import pandas as pd
df = pd.read_csv('listing.csv')
#print(df.head)

class data_cleaning():
    def remove_rows_with_missing_ratings(self):
        df.isna().sum()
        df.dropna(subset=['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating' ,
        'Check-in_rating',  'Value_rating'], inplace= True)
        print(df.head)
        return df

    def combine_description_strings(self):
        pass

cleaned_data = data_cleaning()
cleaned_data.remove_rows_with_missing_ratings()