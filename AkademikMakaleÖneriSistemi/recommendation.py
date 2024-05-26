import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Gerekli NLTK verilerini indir
nltk.download('stopwords')

# İngilizce stopwords listesi
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Metin işleme fonksiyonu
def preprocess(text):
    # Noktalama işaretlerini çıkar
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Küçük harfe çevir
    text = text.lower()
    # Kelime köklerini bul ve stopwords temizle
    text = ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])
    return text

# Veri çerçevesini yükleyelim
df = pd.read_csv('combined_articles.csv')

# Eksik değerleri içeren satırları kaldıralım
df = df.dropna(subset=['abstract'])

# Metin ön işleme
df['processed_abstract'] = df['abstract'].apply(preprocess)

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_abstract'])

# Kosinüs benzerliği matrisini hesaplayalım
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Öneri fonksiyonu
def get_recommendations(title, cosine_sim=cosine_sim):
    # Verilen başlığa göre makalenin indeksini bulalım
    idx = df[df['title'] == title].index[0]

    # Tüm makalelerle olan benzerlik skorlarını alalım
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Benzerlik skorlarına göre makaleleri azalan sırayla sıralayalım
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # İlk 10 benzer makaleyi alalım
    sim_scores = sim_scores[1:11]

    # Benzer makalelerin indekslerini alalım
    article_indices = [i[0] for i in sim_scores]

    # Benzer makaleleri döndürelim
    return df.iloc[article_indices][['title', 'abstract']]

# "human" kelimesini içeren başlıkları listeleyelim
human_titles = df[df['title'].str.contains("human", case=False, na=False)]

if human_titles.empty:
    print("Human ile ilgili başlık bulunamadı.")
else:
    print("Human ile ilgili başlıklar:")
    print(human_titles[['title']])

    # Uygun bir başlık seçelim
    example_title = human_titles['title'].iloc[0]  # İlk başlığı seçiyoruz, bunu değiştirebilirsin

    # Seçtiğimiz başlık için öneriler alalım
    recommendations = get_recommendations(example_title)

    # Önerileri görüntüleyelim
    print(f"Öneriler '{example_title}' için:")
    print(recommendations)
