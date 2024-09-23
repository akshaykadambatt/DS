import pandas as pd
import numpy as np

df = pd.read_csv('./BL-Flickr-Images-Book.csv') #read from file

df.drop(['Edition Statement',
            'Corporate Author',
            'Corporate Contributors',
            'Former owner',
            'Engraver',
            'Contributors',
            'Issuance type',
            'Shelfmarks'], inplace=True, axis=1) #drop columns (axis 1) without returning (inplace)

print(df.head())

print(df['Identifier'].is_unique)

df.set_index('Identifier', inplace=True)

print(df.head())

print(df.iloc[0]) #position based indexing iloc
print(df.loc[206]) #label based indexing loc

print(df.dtypes.value_counts())

extr = df.loc[0:,'Date of Publication'].str.extract(r'^(\d{4})', expand=False)
print(extr.head())
df['Date of Publication'] = pd.to_numeric(extr).fillna(0).astype(int)
print(df['Date of Publication'].dtype)
print(df['Date of Publication'].isnull().sum() / len(df) * 100) #lost data

print(df['Place of Publication'].to_string())

# if contains London, replace with just London, 
# else if Oxford, replace with just Oxford
# else replace - with space
pub = df['Place of Publication']
df['Place of Publication'] = np.where(pub.str.contains('London'), 'London', 
                                      np.where(pub.str.contains('Oxford'),'Oxford', 
                                               pub.str.replace('-', ' ')
                                               )
                                      ) 

print(df['Place of Publication'].head())

print(df.head())

#NEW DATASET

university_towns = []

with open('./university_towns.txt') as file:
    for line in file:
        if '[edit]' in line:
            state = line
        else:
            university_towns.append((state, line))

print(university_towns[:5])

towns_df = pd.DataFrame(university_towns, columns = ['State', 'Town'])
print(towns_df)

def get_citystate(item):
    if ' (' in item:
        return item[:item.find(' (')]
    elif '[' in item:
        return item[:item.find('[')]
    else:
        return item

towns_df = towns_df.map(get_citystate)

print(towns_df)

