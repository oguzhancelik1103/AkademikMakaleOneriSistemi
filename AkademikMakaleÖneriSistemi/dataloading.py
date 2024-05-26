import zipfile
import pandas as pd
import os

# Zip dosyalarının yollarını belirleyelim
inspec_zip_path = 'C:/Users/DELL/Desktop/Inspec.zip'
krapivin_zip_path = 'C:/Users/DELL/Desktop/Krapivin2009.zip'

# Zip dosyalarını açalım ve verileri çıkaralım
with zipfile.ZipFile(inspec_zip_path, 'r') as zip_ref:
    zip_ref.extractall('Inspec')

with zipfile.ZipFile(krapivin_zip_path, 'r') as zip_ref:
    zip_ref.extractall('Krapivin2009')

# Inspec verilerini okuyalım
inspec_docsutf8_folder_path = 'Inspec/Inspec/docsutf8'
inspec_docsutf8_files = os.listdir(inspec_docsutf8_folder_path)

inspec_articles = []

for file_name in inspec_docsutf8_files:
    file_path = os.path.join(inspec_docsutf8_folder_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        lines = content.split('\n')
        title = lines[0]
        abstract = ' '.join(lines[1:])
        inspec_articles.append({
            'title': title,
            'abstract': abstract
        })

inspec_df = pd.DataFrame(inspec_articles)

# Krapivin verilerini okuyalım
krapivin_docsutf8_folder_path = 'Krapivin2009/Krapivin2009/docsutf8'
krapivin_docsutf8_files = os.listdir(krapivin_docsutf8_folder_path)

krapivin_articles = []

for file_name in krapivin_docsutf8_files:
    file_path = os.path.join(krapivin_docsutf8_folder_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        parts = content.split('--')
        title = ''
        abstract = ''
        for part in parts:
            if part.startswith('T'):
                title = part[1:].strip()
            elif part.startswith('A'):
                abstract = part[1:].strip()
        krapivin_articles.append({
            'title': title,
            'abstract': abstract
        })

krapivin_df = pd.DataFrame(krapivin_articles)

# Veri çerçevelerini birleştirelim
combined_df = pd.concat([inspec_df, krapivin_df], ignore_index=True)

# İlk birkaç satırı inceleyelim
print(combined_df.head())

# Veri çerçevesini bir dosyaya kaydedelim
combined_df.to_csv('combined_articles.csv', index=False)
