import pandas as pd
import regex as re
import os
from sklearn.preprocessing import MinMaxScaler


#*******************************UTILIZA ESTA RUTA COMO FICHERO INICIAL PARA LIMPIEZA: *************************************

# Define the file path of the CSV file
file_path = r'..\new_dataset\all_data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)


#******************************* AÑADE ESTO ANTES DE LIMPIAR FORMATOS:*****************************************************

def to_lowercase(text):
    return text.lower()

def normalize_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def strip_trailing_whitespaces(text):
    return text.strip()

# We start cleaning Formats, using previously defined functions. We put everything to lowercase,
# strip whitespaces and also check that there are no double spaces
df['Formats'] = df['Formats'].apply(to_lowercase)
df['Formats'] = df['Formats'].apply(strip_trailing_whitespaces)
df['Formats'] = df['Formats'].apply(normalize_spaces)
df['Formats'] = df['Formats'].str.replace('.', ',') 


#**************************************ANADE ESTO DESPUÉS DE LIMPIEZA DE FROMATOS**************************************************************************


# We initialize a dictionary to map product names to IDs
product_id_map = {}
current_id = 1

def get_product_id(product_name):
    global current_id
    if product_name not in product_id_map:
        product_id_map[product_name] = current_id
        current_id += 1
    return product_id_map[product_name]



# We apply the function to create a new 'product_id' column
df['product_id'] = df['Name'].apply(get_product_id)

# We create the 'Pack' column
df['Pack'] = df['Formats'].str.contains('pack', case=False)


# Function to determine the package value
def find_package(format_str):
    if 'bolsa' in format_str:
        return 'bolsa'
    elif 'saco' in format_str:
        return 'saco'
    elif 'lata' in format_str:
        return 'lata'
    elif 'latas' in format_str:
        return 'lata'
    elif 'saco' in format_str:
        return 'saco'
    elif 'sobre' in format_str:
        return 'sobre'
    elif 'pouches' in format_str:
        return 'pouch'
    else:
        return "not_specified"

# Applying the function to create the 'package' column
df['Package'] = df['Formats'].apply(find_package)



# We normalize numerical variables using the MinMaxScaler
scaler = MinMaxScaler()

# We standardize numerical columns amd save values in columns named st_column_name
df['st_price'] = scaler.fit_transform(df[['Price']])

df['st_kg_price'] = scaler.fit_transform(df[['Weight']])

df['st_star_ratings'] = scaler.fit_transform(df[['Weight']])

df['st_weight'] = scaler.fit_transform(df[['Weight']])

#**********************TODO A CATEGORICOS AL FINAL, PORQUE AQUÍ EXISTE PACKEGE, ANTES NO****************
df['Formats'] = df['Formats'].astype('category')
df['Name'] = df['Name'].astype('category')
df['Formats'] = df['Formats'].astype('category')
df['animal'] = df['animal'].astype('category')
df['food_type'] = df['food_type'].astype('category')
df['subcategory'] = df['subcategory'].astype('category')
df['Package'] = df['Package'].astype('category')



#***************HE QUITADO COLUMNAS QUE NO UTILIZAMOS, NO SE SI ES MEJOR DEJARLAS O BORRARLAS*****
# We remove unnecessary columns:
df.drop('Name', axis=1, inplace=True)
df.drop('Description', axis=1, inplace=True)
df.drop('Formats', axis=1, inplace=True)







#-----------------------------------------------------------------------------------------------------------

# We specify the full path of the directory where we want to save the file
directory = r'..\clean_dataset_ada'

# We specify the file name (e.g., 'all_data.csv')
file_name = 'clean_data.csv'

# We concatenate the directory path and the file name
file_path = os.path.join(directory, file_name)

# We save the clean DataFrame to a CSV file in the specified directory
df.to_csv(file_path, index=False)