!pip install --user tensorflow
!pip install --user keras
! pip install pydrive
!pip install nltk
!pip install numpy
!pip install pickle-mixin
!pip install scipy
!pip install scikit-learn
!pip install bahasa
!pip3 install rasa
! pip install gensim
!pip install fasttext
!sudo python setup.py install


!wget "https://www.dropbox.com/s/9vabe1vci7cnt57/id.tar.gz?dl=0" -O corpus.txt

!wget "https://drive.google.com/u/0/uc?id=1A8pop92mV3XQI6mhRHIcJK_8kxQQG_Ar&export=download" -O corpus.txt

from gensim.models import FastText
import os

# Ganti dengan path ke file teks berbahasa Indonesia
corpus_path = "path_to_your_corpus.txt"

# Baca teks dari file corpus
corpus = []
with open("corpus.txt", "r", encoding="utf-8") as file:
    corpus = file.readlines()

# Pra-pelatihan model FastText dengan data teks berbahasa Indonesia
model = FastText(size=300, window=5, min_count=5, workers=os.cpu_count())
model.build_vocab(sentences=corpus)
model.train(sentences=corpus, total_examples=len(corpus), epochs=10)

# Simpan model pra-pelatihan ke dalam file
model.save("pretrained_fasttext_id.model")




import fasttext
model = fasttext.train_unsupervised('wiki-id-formatted.txt', model='skipgram')

model300 = fasttext.train_unsupervised('wiki-id-formatted.txt', model='skipgram', dim=300)
model.save_model("trained_model_id.bin")
vektor = model["jakarta"]<br>print(len(vektor))



import math
def vector_len(v):
    return math.sqrt(sum([x*x for x in v]))
def dot_product(v1, v2):
    return sum([x*y for (x,y) in zip(v1, v2)])
def cosine_similarity(v1, v2):
    return dot_product(v1, v2) / (vector_len(v1) * vector_len(v2))
def most_similar(x, top=10):
    v1 = model.get_word_vector(x)
    all_word = []
    for word in model.words:
        if word!=x:
            v2 = model.get_word_vector(word)
            all_word.append((cosine_similarity(v1,v2), word))
 most_similar("bandung")