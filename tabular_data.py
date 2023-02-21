import pandas as pd
#df = pd.read_csv('listing.csv')
#print(df.head)

#class data_cleaning():
def remove_rows_with_missing_ratings(): 
    df.isna().sum()
    df.dropna(subset=['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating' ,
    'Check-in_rating',  'Value_rating'], inplace= True)
    print(df.head)
    return df

def combine_description_strings():
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
    

def set_default_feature_values():
    df[['bathrooms', 'beds', 'guests', 'bedrooms']].fillna(1, inplace = True)
    # df['beds'].fillna(1)
    # df['guests'].fillna(1)
    # df['bedrooms'].fillna(1, inplace= True)
    return df
    

def clean_tabular_data(df):
    remove_rows_with_missing_ratings()
    combine_description_strings()
    set_default_feature_values()
    return df

if __name__ == "__main__":
    df = pd.read_csv('listing.csv')
    cleaned_data = clean_tabular_data(df)
    cleaned_data.to_csv('clean_tabular_data.csv')
