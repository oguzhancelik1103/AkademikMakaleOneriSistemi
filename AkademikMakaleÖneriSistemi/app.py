from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

app = Flask(__name__)
app.secret_key = 'erkanenaktarlarkoltuginaltindakalikbeniara89'

# İngilizce durak kelimeleri indiriliyor ve stopwords seti oluşturuluyor.
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Metin ön işleme fonksiyonu: noktalama işaretleri kaldırılır, küçük harfe dönüştürülür ve kökleri bulunur.
def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])

# MongoDB bağlantısı kuruluyor.
client = MongoClient('mongodb://localhost:27017/')
db = client['AkademikMakaleOneriSistemi']
users_collection = db['Users']

# Makale verileri okunuyor ve başlıklar ön işlemden geçiriliyor.
df = pd.read_csv('cleaned_combined_articles.csv').dropna(subset=['title'])
df['processed_title'] = df['title'].apply(preprocess)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_title'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Kullanıcının okuma geçmişini güncelleyen fonksiyon.
def update_reading_history(user_email, recommended_articles):
    users_collection.update_one(
        {"email": user_email},
        {"$push": {"reading_history": {"$each": recommended_articles}}}
    )

# Kişiselleştirilmiş makale önerileri getiren fonksiyon.
def get_personalized_recommendations(user_email, search_query=""):
    user = users_collection.find_one({"email": user_email})
    if not user:
        return pd.DataFrame()

    interests_weighted = user.get('interests', []) * 1
    search_query_weighted = [search_query] * 8
    reading_history_weighted = user.get('reading_history', []) * 1

    combined_input = interests_weighted + search_query_weighted + reading_history_weighted
    processed_input = [preprocess(text) for text in combined_input]
    input_vector = vectorizer.transform([' '.join(processed_input)])
    
    sim_scores = list(enumerate(cosine_similarity(input_vector, tfidf_matrix).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:50]
    
    seen_titles = set()
    unique_articles = []
    for index, score in sim_scores:
        title = df.iloc[index]['title']
        if title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(index)
        if len(unique_articles) == 10:
            break

    recommended_articles = df.iloc[unique_articles]['title'].tolist()
    update_reading_history(user_email, recommended_articles)
    return df.iloc[unique_articles]

# Rotalar ve ilgili fonksiyonlar tanımlanıyor.
@app.route('/recommend', methods=['POST'])
def recommend():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    search_query = request.form.get('search_query', '')
    recommendations = get_personalized_recommendations(session['email'], search_query)
    
    if recommendations.empty:
        return render_template('recommendations.html', articles=[], message="No recommendations found based on your interests.")
    
    return render_template('recommendations.html', articles=recommendations.to_dict('records'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        interests = request.form.getlist('interests')
        if users_collection.find_one({"email": email}):
            return 'Email already exists'
        users_collection.insert_one({"email": email, "password": generate_password_hash(password, 'pbkdf2:sha256'), "interests": interests, "reading_history": []})
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users_collection.find_one({"email": email})
        if user and check_password_hash(user['password'], password):
            session['email'] = email
            return redirect(url_for('dashboard'))
        return 'Invalid email or password'
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'email' not in session:
        return redirect(url_for('login'))
    user = users_collection.find_one({"email": session['email']})
    return render_template('dashboard.html', interests=user.get('interests', 'Not set'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'email' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        name = request.form['name']
        surname = request.form['surname']
        email = request.form['email']
        interests = request.form.getlist('interests')
        users_collection.update_one({"email": session['email']}, {"$set": {"name": name, "surname": surname, "email": email, "interests": interests}})
        session['email'] = email
        return redirect(url_for('dashboard'))
    user = users_collection.find_one({"email": session['email']})
    return render_template('profile.html', user=user)

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
