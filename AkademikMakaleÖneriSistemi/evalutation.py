import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
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
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])
    return text

# Temizlenmiş veri çerçevesini yükleyelim
df = pd.read_csv('cleaned_combined_articles.csv')

# Eksik değerleri içeren satırları kaldıralım
df = df.dropna(subset=['title'])

# Metin ön işleme
df['processed_title'] = df['title'].apply(preprocess)

# Eğitim ve test verisi olarak ayıralım
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Veri çerçevelerinin boyutlarını kontrol edelim
print(f"Toplam veri sayısı: {len(df)}")
print(f"Eğitim verisi sayısı: {len(train_df)}")
print(f"Test verisi sayısı: {len(test_df)}")

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer()
tfidf_matrix_train = vectorizer.fit_transform(train_df['processed_title'])
tfidf_matrix_test = vectorizer.transform(test_df['processed_title'])

# Kosinüs benzerliği matrisini hesaplayalım
cosine_sim_train = cosine_similarity(tfidf_matrix_train, tfidf_matrix_train)
cosine_sim_test = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)

# Benzerlik matrislerinin boyutlarını kontrol edelim
print(f"TF-IDF eğitim matrisi boyutu: {tfidf_matrix_train.shape}")
print(f"TF-IDF test matrisi boyutu: {tfidf_matrix_test.shape}")
print(f"Kosinüs benzerliği eğitim matrisi boyutu: {cosine_sim_train.shape}")
print(f"Kosinüs benzerliği test matrisi boyutu: {cosine_sim_test.shape}")

# Eğitim veri setinde olmayan test başlıklarını filtreleme
test_titles = test_df['title'].tolist()
valid_test_titles = [title for title in test_titles if title in train_df['title'].values]
invalid_test_titles = [title for title in test_titles if title not in train_df['title'].values]

print(f"Eğitim veri setinde olmayan test başlıkları sayısı: {len(invalid_test_titles)}")
print(f"Eğitim veri setinde olan test başlıkları sayısı: {len(valid_test_titles)}")

# Eğitim veri setinde olmayan başlıkları yazdırma
print("Eğitim veri setinde olmayan test başlıkları:")
for title in invalid_test_titles[:5]:  # İlk 5 başlık
    print(title)

# Geçerli test başlıkları için dataframe oluşturma
valid_test_df = test_df[test_df['title'].isin(valid_test_titles)]

# Öneri fonksiyonu (eğitim verisi için)
def get_recommendations(title, df, cosine_sim):
    try:
        idx = df[df['title'] == title].index[0]
    except IndexError:
        print(f"Başlık bulunamadı: {title}")
        return pd.DataFrame()

    if idx >= cosine_sim.shape[0]:
        print(f"İndeks {idx} geçersiz, benzerlik matrisinin sınırlarının dışında")
        return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    valid_sim_scores = [score for score in sim_scores if score[0] < len(df)]

    article_indices = [i[0] for i in valid_sim_scores]

    if not article_indices:
        print("Geçerli bir benzer makale bulunamadı.")
        return pd.DataFrame()

    return df.iloc[article_indices][['title']]

# Değerlendirme (eğitim verisi)
def evaluate_recommendations(df, cosine_sim):
    test_titles = df['title'].tolist()
    for title in test_titles[:5]:  # İlk 5 başlık için değerlendirme yapıyoruz
        print(f"\nBaşlık: {title}")
        recommendations = get_recommendations(title, df, cosine_sim)
        if not recommendations.empty:
            print(f"Öneriler '{title}' için:")
            print(recommendations)
        else:
            print("Geçerli öneri bulunamadı.")

# Değerlendirme süreci (eğitim verisi)
print("Eğitim verisi değerlendirmesi:")
evaluate_recommendations(train_df, cosine_sim_train)

# Test verisi için öneri fonksiyonu (eğitim verisine göre)
def get_test_recommendations(title, test_df, train_df, cosine_sim_test):
    try:
        idx = test_df[test_df['title'] == title].index[0]
    except IndexError:
        print(f"Başlık bulunamadı: {title}")
        return pd.DataFrame()

    if idx >= cosine_sim_test.shape[0]:
        print(f"İndeks {idx} geçersiz, benzerlik matrisinin sınırlarının dışında")
        return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim_test[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    valid_sim_scores = [score for score in sim_scores if score[0] < len(train_df)]

    article_indices = [i[0] for i in valid_sim_scores]

    if not article_indices:
        print("Geçerli bir benzer makale bulunamadı.")
        return pd.DataFrame()

    return train_df.iloc[article_indices][['title']]

# Değerlendirme süreci (test verisi)
print("\nTest verisi değerlendirmesi:")
for title in valid_test_titles[:5]:  # İlk 5 başlık için değerlendirme yapıyoruz
    print(f"\nBaşlık: {title}")
    recommendations = get_test_recommendations(title, valid_test_df, train_df, cosine_sim_test)
    if not recommendations.empty:
        print(f"Öneriler '{title}' için:")
        print(recommendations)
    else:
        print("Geçerli öneri bulunamadı.")
