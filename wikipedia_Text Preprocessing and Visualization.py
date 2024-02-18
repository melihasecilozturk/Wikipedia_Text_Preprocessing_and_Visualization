#################################################
# WIKI 1 - Metin Ön işleme ve Görselleştirme (NLP - Text Preprocessing & Text Visualization)
#################################################

###################f##############################
# Problemin Tanımı
#################################################
# Wikipedia örnek datasından metin ön işleme, temizleme işlemleri gerçekleştirip, görselleştirmeleri yapmak.

#################################################
# Veri Seti Hikayesi
#################################################


#################################################
# Gerekli Kütüphaneler ve ayarlar



import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from warnings import filterwarnings


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)

# Datayı okumak
df = pd.read_csv("/Users/melihasecilozturk/Desktop/miuul/natural_language_processing/NLP_wiki-221126-161428/wiki_data.csv", index_col=0)
df.head()
df = df[:2000]

df.head()
df.shape

#################################################
# Görevler
#################################################
#Adım 1: Metin ön işleme için clean_text adında fonksiyon oluşturunuz. Fonksiyon;
# • Büyük küçük harf dönüşümü,
# • Noktalama işaretlerini çıkarma,
# • Numerik ifadeleri çıkarma Işlemlerini gerçekleştirmeli.


def clean_text(text):
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace(r'[^\w\s]', '')
    text = text.str.replace("\n" , '')
    # Numbers
    text = text.str.replace('\d', '')
    return text

#Adım 2: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.

df["text"] = clean_text(df["text"])

df.head()



# Adım 3: Metin içinde öznitelik çıkarımı yaparken önemli olmayan kelimeleri çıkaracak remove_stopwords adında fonksiyon
# yazınız.

def remove_stopwords(text):
    stop_words = stopwords.words('English')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    return text

# Adım 4: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.

df["text"] = remove_stopwords(df["text"])




# Adım 5: Metinde az geçen (1000'den az, 2000'den az gibi) kelimeleri bulunuz. Ve bu kelimeleri metin içerisinden çıkartınız.

pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]

sil = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))




# Adım 6: Metinleri tokenize edip sonuçları gözlemleyiniz.

df["text"].apply(lambda x: TextBlob(x).words)


# Adım 7: Lemmatization işlemi yapınız.
# ran, runs, running -> run (normalleştirme)

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df.head()



# Görev 2: Veriyi Görselleştiriniz

# Adım 1: Metindeki terimlerin frekanslarını hesaplayınız.

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index() # kodu güncellemek gerekecek

tf.head()

# Adım 2: Bir önceki adımda bulduğunuz terim frekanslarının Barplot grafiğini oluşturunuz.

# Sütunların isimlendirilmesi
tf.columns = ["words", "tf"]
# 5000'den fazla geçen kelimelerin görselleştirilmesi
tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
plt.show()

# Adım 3: Kelimeleri WordCloud ile görselleştiriniz.

# kelimeleri birleştirdik
text = " ".join(i for i in df["text"])

# wordcloud gÃ¶rselleÅŸtirmenin Ã¶zelliklerini belirliyoruz
wordcloud = WordCloud(max_font_size=50,
max_words=100,
background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Görev 3: Tüm Aşamaları Tek Bir Fonksiyon Olarak Yazınız
# Adım 1: Metin ön işleme işlemlerini gerçekleştiriniz.
# Adım 2: Görselleştirme işlemlerini fonksiyona argüman olarak ekleyiniz.
# Adım 3: Fonksiyonu açıklayan 'docstring' yazınız.

df = pd.read_csv("/Users/melihasecilozturk/Desktop/miuul/natural_language_processing/NLP_wiki-221126-161428/wiki_data.csv", index_col=0)


def wiki_preprocess(text, Barplot=False, Wordcloud=False):
    """
    Textler üzerinde ön işleme işlemleri yapar.

    :param text: DataFrame'deki textlerin olduğu değişken
    :param Barplot: Barplot görselleştirme
    :param Wordcloud: Wordcloud görselleştirme
    :return: text


    Example:
            wiki_preprocess(dataframe[col_name])

    """
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace("\n", '')
    # Numbers
    text = text.str.replace('\d', '')
    # Stopwords
    sw = stopwords.words('English')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    # Rarewords / Custom Words
    sil = pd.Series(' '.join(text).split()).value_counts()[-1000:]
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in sil))


    if Barplot:
        # Terim Frekanslarının hesaplanması
        tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        # Sütunların isimlendirilmesi
        tf.columns = ["words", "tf"]
        # 5000'den fazla geçen kelimelerin görselleştirilmesi
        tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
        plt.show()

    if Wordcloud:
        # Kelimeleri birleştirdik
        text = " ".join(i for i in text)
        # wordcloud görselleştirmelerinin özelliklerini belirliyoruz
        wordcloud = WordCloud(max_font_size=50,
                              max_words=100,
                              background_color="white").generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    return text

wiki_preprocess(df["text"])

wiki_preprocess(df["text"], True, True)