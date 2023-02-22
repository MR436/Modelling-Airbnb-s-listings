import pandas as pd
import typing
#df = pd.read_csv('listing.csv')
#print(df.head)

#class data_cleaning():
def remove_rows_with_missing_ratings(df): 
    df.isna().sum()
    df.dropna(subset=['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating' ,
    'Check-in_rating',  'Value_rating', 'amenities_count'], inplace= True)
    #print(df.head)
    return df

def combine_description_strings(df):
    df.dropna(subset='Description', inplace=True)
    df['Description'] = df.Description.str.split(',')
    for i in df.Description:
        i.remove(i[0])
        #print(i)
    for i in df.Description:
            #print(i[0])
        for j in i:
            j.strip()
            " ".join(j.split(' '))
            j.replace('\n','')
            j.replace('','.')
            #print(type(j))
            #print(j)
    

def set_default_feature_values(df):
    df = df[['bathrooms', 'beds', 'guests', 'bedrooms']].fillna(1, inplace = True)
    
    # df['beds'].fillna(1)
    # df['guests'].fillna(1)
    # df['bedrooms'].fillna(1, inplace= True)
    return df
    

def clean_tabular_data(df):
    remove_rows_with_missing_ratings(df)
    combine_description_strings(df)
    set_default_feature_values(df)
    return df

def load_airbnb(label : str):
    df = pd.read_csv('listing.csv')
    cleaned_data = clean_tabular_data(df)
    #cleaned_data.to_csv('clean_tabular_data.csv')
    target = cleaned_data[label]
    features = cleaned_data.drop(columns = label)
    features_num = features.select_dtypes(include = [int, float])

    return (features_num, target)
     


if __name__ == "__main__":
    
    load_airbnb('Price_Night')
    


    #print(features.head(5))

 