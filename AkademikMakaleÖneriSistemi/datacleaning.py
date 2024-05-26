import pandas as pd

# Veri çerçevesini yükleyelim
df = pd.read_csv('combined_articles.csv')

# Bir satırda birden fazla başlık varsa ayırma işlemi
def split_multiple_titles(row):
    if isinstance(row['title'], str):
        titles = row['title'].split('\n')
        
        # Her başlığı ayrı bir satıra yerleştir
        rows = []
        for title in titles:
            rows.append({'title': title.strip()})
        return rows
    else:
        return [{'title': row['title']}]

# Yeni bir DataFrame oluşturmak için tüm satırları işleyelim
new_rows = []
for _, row in df.iterrows():
    new_rows.extend(split_multiple_titles(row))

cleaned_df = pd.DataFrame(new_rows)

# "R" ile başlayan başlıkları temizleyelim
cleaned_df = cleaned_df[~cleaned_df['title'].str.startswith('R', na=False)]

# Temizlenmiş veri setini kaydedelim
cleaned_df.to_csv('cleaned_combined_articles.csv', index=False)

print("Veri seti temizlendi ve 'cleaned_combined_articles.csv' dosyasına kaydedildi.")
