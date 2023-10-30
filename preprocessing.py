#!/usr/bin/env python
# coding: utf-8

# In[5]:


# 등장 빈도 기준 정제 함수 정의 
def clean_by_freq(tokenized_words, freq):
    from collections import Counter
    # 1. Counter 함수를 통해 단어의 빈도수를 카운트하여 단어 집합 생성
    # vocab 이라는 변수에 담기
    vocab = Counter(tokenized_words)
    
    # 2. 빈도수가 freq 이하인 단어 추출
    # low_freq_words라는 변수에 담기 
    low_freq_words = []

    for key, value in vocab.items():
        if value <= freq:
            low_freq_words.append(key)
    
    # 3. low_freq_words에 포함되지 않는 단어 리스트 생성 (not in)
    # cleaned_words 라는 변수에 담기 
    cleaned_words = []

    for word in tokenized_words:
        if word not in low_freq_words:
            cleaned_words.append(word)
    
    return cleaned_words

# 단어 길이 기준 정제 함수 
def clean_by_len(tokenized_words, length):
    
    cleaned_by_freq_len = []
    
    # 단어길이가 length이상인 단어들을 cleaned_by_freq_len 담아주기
    for word in tokenized_words:
        if len(word) >= length:
            cleaned_by_freq_len.append(word)
    
    return cleaned_by_freq_len


# In[6]:


# 불용어 제거 함수 만들기
def clean_by_stopwords(tokenized_words, stop_words_set):
    
    cleand_words = []
    
    # 불용어 제거해주는 코드 작성
    for word in tokenized_words:
        if word not in stop_words_set:
            cleand_words.append(word)
    
    
    return cleand_words


# In[7]:


from nltk.stem import PorterStemmer

# 포터스테머 어간 추출함수
def stemming_by_porter(tokenized_words):
    porter_stemmer = PorterStemmer()
    porter_stemmed_words = []
    
    for word in tokenized_words:
        stem = porter_stemmer.stem(word)
        porter_stemmed_words.append(stem)
    
    return porter_stemmed_words


# In[8]:


from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag

# 품사 태깅 함수 정의
def pos_tagger(tokenized_sents):
    pos_tagged_words = []
    # 단어 토큰화  - word_tokenize
    for sentence in tokenized_sents:
        tokenized_words = word_tokenize(sentence)
    
    # 품사 태깅 - pos_tag
        pos_tagged = pos_tag(tokenized_words)
        
    # 품사태깅한 데이터 담아주기 - extend
        pos_tagged_words.extend(pos_tagged)
    
    return pos_tagged_words


# In[9]:


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('omw-1.4')

# pennTree -> WordNet 로 변환
def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    else:
        return
    
def words_lemmatizer(pos_tagged_words):  # 표제어를 추출해주는 함수 정의
        lemmatizer = WordNetLemmatizer()  # 객체 생성
        lemmatized_words = []  # 표제어 추출된 단어를 담는 리스트
        
        for word, tag in pos_tagged_words:
            wn_tag = penn_to_wn(tag)
            
            if wn_tag in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
                lemmatized_words.append(lemmatizer.lemmatize(word, wn_tag))
            else:
                lemmatized_words.append(word)
        return lemmatized_words
        


# In[ ]:




