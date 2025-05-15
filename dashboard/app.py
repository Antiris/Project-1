import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from pymorphy3 import MorphAnalyzer

# NLTK
nltk.download('stopwords')
nltk.download('punkt')

import streamlit as st

st.set_page_config(page_title="Анализ песен Pyrokinesis", layout="wide")

st.title("Исследование лирики Pyrokinesis")
st.markdown("""
Анализ текстов песен Pyrokinesis на основе данных с Яндекс Музыки.
""")

@st.cache_data
def load_data():
    df = pd.read_csv('../data/processed/pyrokinesis_after.csv')
    return df.dropna(subset=['lyrics'])

df = load_data()

tab1, tab2, tab3, tab4 = st.tabs([
    "Общая статистика", 
    "Повторяющиеся образы", 
    "Тематическая кластеризация",
    "Выводы"
])

with tab1:
    st.subheader("Годы релиза")
    fig, ax = plt.subplots()
    sns.histplot(df['release_year'], bins=20, kde=False, color='skyblue', ax=ax)
    ax.set_title('Распределение треков по годам релиза')
    st.pyplot(fig)

    st.subheader("Топ-альбомы по числу песен")
    top_albums = df['album'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_albums.values, y=top_albums.index, ax=ax)
    ax.set_title('Топ-10 альбомов')
    st.pyplot(fig)

    st.subheader("Длина текста песен (в словах)")
    df['lyrics_wordcount'] = df['lyrics'].apply(lambda x: len(str(x).split()))
    fig, ax = plt.subplots()
    sns.histplot(df['lyrics_wordcount'], bins=20, color='salmon', ax=ax)
    ax.set_title('Распределение количества слов в тексте песен')
    st.pyplot(fig)

    st.write("**Статистика:**")
    st.dataframe(df['lyrics_wordcount'].describe().to_frame())

russian_stopwords = stopwords.words('russian')
morph = MorphAnalyzer()

excluded_lemmas = {
    'мой', 'твой', 'свой', 'наш', 'ваш', 'их', 'его', 'её', 'ее',
    'тот', 'этот', 'такой', 'таков', 'всякий', 'любой', 'некоторый', 'самый',
    'один', 'другой', 'каждый', 'иной', 'весь', 'всё', 'все', 'никакой'
}

def lemmatize_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in russian_stopwords]
    
    lemmas = []
    for token in tokens:
        parsed = morph.parse(token)[0]
        pos = parsed.tag.POS
        lemma = parsed.normal_form
        if (
            pos not in {'PREP', 'PRCL', 'CONJ', 'NPRO', 'INTJ'} and
            lemma not in excluded_lemmas
        ):
            lemmas.append(lemma)
    
    return ' '.join(lemmas)

with tab2:
    st.header("Облако слов из текстов Pyrokinesis")

    with st.spinner("Загрузка и обработка данных..."):
        df = pd.read_csv('../data/processed/pyrokinesis.csv')
        df = df.dropna(subset=['lyrics'])
        df['lemmatized_lyrics'] = df['lyrics'].apply(lemmatize_text)

        vectorizer = CountVectorizer(
            token_pattern=r'\b[а-яА-ЯёЁ]{2,}\b'
        )
        X = vectorizer.fit_transform(df['lemmatized_lyrics'])
        word_freq = X.toarray().sum(axis=0)
        vocab = vectorizer.get_feature_names_out()
        freq_dict = dict(zip(vocab, word_freq))

        wordcloud = WordCloud(
            font_path='C:/Windows/Fonts/arial.ttf',
            width=1000,
            height=500,
            background_color='white'
        ).generate_from_frequencies(freq_dict)

    st.subheader("Облако слов (лемматизированные тексты)")
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    st.subheader("Корреляция топ-30 слов")

    df_words = pd.DataFrame(X.toarray(), columns=vocab)
    corr_matrix = df_words.corr()

    top_words = sorted(freq_dict, key=freq_dict.get, reverse=True)[:30]
    top_corr = corr_matrix.loc[top_words, top_words]

    fig_corr, ax_corr = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        top_corr,
        annot=False,
        cmap='coolwarm',
        xticklabels=True,
        yticklabels=True,
        ax=ax_corr
    )
    ax_corr.set_title('Корреляция между топ-30 словами в песнях Pyrokinesis', fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig_corr)

    

with tab3:
    st.subheader("Кластеризация песен (TF-IDF + KMeans + PCA)")

    if 'lemmatized_lyrics' not in df.columns:
        st.warning("Нет лемматизированных данных.")
    else:
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.8, token_pattern=r'\b[а-яё]{3,}\b')
        X = vectorizer.fit_transform(df['lemmatized_lyrics'])

        n_clusters = st.slider("Количество кластеров", 2, 15, 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)

        df['cluster'] = kmeans.labels_

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X.toarray())

        fig, ax = plt.subplots(figsize=(10, 7))
        for cluster_num in range(n_clusters):
            ax.scatter(
                X_pca[df['cluster'] == cluster_num, 0],
                X_pca[df['cluster'] == cluster_num, 1],
                label=f'Кластер {cluster_num}'
            )
        ax.legend()
        ax.set_title('Кластеризация песен Pyrokinesis')
        st.pyplot(fig)

        st.subheader("Песни по кластерам")
        selected_cluster = st.selectbox("Выберите кластер:", sorted(df['cluster'].unique()))
        st.dataframe(df[df['cluster'] == selected_cluster][['title', 'album', 'release_year']].head(10))

with tab4:
    st.subheader("Выводы")

    st.markdown("""
    **Подтверждённые гипотезы:**
    - В песнях Pyrokinesis часто встречаются повторы образов (смерть, любовь, небо, бог, пустота)
    - В текстах наблюдается тематическая связь, которое можно разбить на смысловые кластеры

    **Что сделано:**
    - Очистка и лемматизация текстов
    - Построение облака слов
    - Корреляционный анализ
    - Кластеризация с интерпретацией

    **Что можно улучшить:**
    - Добавить тематическое моделирование (LDA)
    - Провести временной анализ по годам

    **Кому может быть интересно:**
    - Музыкальным критикам, фанатам Pyrokinesis
    - Лингвистам и исследователям культуры
    - Студентам, изучающим NLP на русском языке
    """)