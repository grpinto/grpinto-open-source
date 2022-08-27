import pandas as pd
import numpy as np
import sys
import nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import tokenize
import string
import re
import random
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_iris
from nltk.tokenize import PunktSentenceTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer

# all words from subtitles tokenized and lemmatized


def central_function(file):

    sub1 = open(file).read().strip().split('\n')
# sub2 = open("/Users/kingsize/Desktop/MrBeanLegendas.srt").read().strip().split('\n')
# sub3 = open("/Users/kingsize/Desktop/TheDescentLegendas.srt").read().strip().split('\n')
    sub = open(file).read()
    line_lemm = worked_words(sub)
    data_text = organizing_data(line_lemm)
    all_sent_sia = sia_attribution(data_text)
    all_sent_sia_polarity = tb_polarity(all_sent_sia)
    my_graph = graphdraw(all_sent_sia_polarity)
    got_chunk = chunk(sub1)
    tagging = create_tagged_words(got_chunk)
    print(all_sent_sia_polarity, '\n\n')
    #print('\n\n', tagging, '\n\n')
    average = calculate_average(line_lemm)
    return all_sent_sia_polarity


def worked_words(sub):
    sub_words = word_tokenize(sub)
    stop_words = stopwords.words("english")
    punct = list(string.punctuation)

# melhorar a analise deixando estar por exemplo ponto de exclamação

    stops = stop_words + punct + ["''", 'r.', '``', "'s", "n't"]

    filtered_words = []
    for w in sub_words:
        if w.lower() not in stops:
            filtered_words.append(w.lower())

    lem = WordNetLemmatizer()
    lemm_words = []
    for w in filtered_words:
        lemm_words.append(lem.lemmatize(w))

    # print('There are', len(lemm_words), 'words total')
    # print('There are', len(np.unique(lemm_words)), 'unique words')

    lemm_col = pd.Series(lemm_words)

    word_counts = lemm_col.value_counts()
    solo = word_counts[word_counts < 2].index.values.tolist()

    remove = stops + solo

    # creating a list of cleaned strings for the words on each line

    lines = pd.Series(re.split('\n\n', sub))
    # print("There are", len(lines) - 1, "lines")
    time_frame = []
    line_lemm = []
    for i in range(len(lines)):
        text = lines[i]
        # print(text)
        tokens = RegexpTokenizer(r'[a-zA-Z\']+').tokenize(text)
        final = []
        for w in tokens:
            if w.lower() not in remove:
                final.append(w.lower())
        lemm_word = []
        for w in final:
            lemm_word.append(lem.lemmatize(w))
        line_lemm.append(' '.join(lemm_word))
    # print(time_frame)
    return line_lemm

# Stemming


def stemmed_words(lemm_words):
    stemLine = []
    porter = PorterStemmer()
    for i in lemm_words:
        stemLine.append(porter.stem(i))
    return stemLine

# Bag of words


def create_bag(line_lemm):
    bow = CountVectorizer()
    BOW = bow.fit_transform(line_lemm)
    bagOFwords = pd.DataFrame(BOW.toarray())
    bagOFwords.columns = bow.get_feature_names_out()
    bagOFwords.head()
    return bagOFwords

# Bow counts and exploration


def bag_analysis(line_lemm):
    bagofwords = create_bag(line_lemm)
    count_word = bagofwords.mean(axis=0)
    small = count_word[count_word == 0].index.values.tolist()
    final_bag = bagofwords.drop(small, axis=1)
    print('\nDimensions of Array : \n', final_bag.shape)
    print('\nTop 10 Average Count Bag of Words\n', final_bag.mean(axis=0).sort_values(ascending=False)[0:10])
    top_10 = final_bag.mean(axis=0).sort_values(ascending=False)[0:10].index.values.tolist()
    return final_bag.head(), top_10

# TF-IDF


def create_tfidf(line_lemm):
    vectorizer = TfidfVectorizer()
    got_tfidf = vectorizer.fit_transform(line_lemm)
    tfidf = pd.DataFrame(got_tfidf.toarray())
    tfidf.columns = vectorizer.get_feature_names_out()
    return tfidf

# TF-IDF counts and exploration


def tfidf_analysis(line_lemm, sub1, top_10):
    tfidf = create_tfidf(line_lemm)
    avg_tfidf = tfidf.mean(axis=0)
    print('\nTF-IDF Scores for Bag of words top 10\n', avg_tfidf[top_10])
    print('\nTop 10 Highest Average TF-IDF Scores\n', avg_tfidf.sort_values(ascending=False)[0:10])
    document = ' '.join(sub1[8:10]) # arbitrary limits
    return document


def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


def post_preprocess(document, sub1):
    sent = preprocess(document)
    print('\n', sub1[8:10])
    print(sent)
    return sent

# chunking process


def chunk(sub1):
    document2 = ' '.join(sub1[8:10])
    big_sent = preprocess(document2)
    return big_sent

# created earlier-POS tagged words


def create_tagged_words(big_sent):
    pattern = 'NP: {<DT>?<JJ>*<NN.?>+}'
    pattern2 = 'Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}'
    pattern3 = 'Chunk: {<NN.?>+<RB.?>*<VB.?>*<RB.?>*<JJ.?>*<CC>?}'
    # NP: {<DT>? <JJ>* <NN>*} # NP
    # P: {<IN>}           # Preposition
    # V: {<V.*>}          # Verb
    # PP: {<P> <NP>}      # PP -> P NP
    # VP: {<V> <NP|PP>*}  # VP -> V (NP|PP)*
    cp = nltk.RegexpParser(pattern2)
    cs = cp.parse(big_sent)
    #print('\n', cs)
    return cs


def organizing_data(line_lemm):
    all_sent = list()
    for i in range(1, len(line_lemm)):
        temp = pd.DataFrame()
        line = line_lemm[i]
    # we can change back to the original by putting lines
        temp['sentence'] = tokenize.sent_tokenize(line)
        temp['line'] = i
        all_sent.append(temp)

    all_sent = pd.concat(all_sent, ignore_index=True)
    # all_sent.head()
    return all_sent


# sentences = tokenize.sent_tokenize(sub[90:985])


# nltk sentiment analysis

# sia = SentimentIntensityAnalyzer()
# for sentence in sentences:
#    print(sentence)
#    ss = sia.polarity_scores(sentence)
#    for k in sorted(ss):
#        print('{0}: {1}, '.format(k, ss[k]), end='')
#    print()

# applying to all sentences


def sia_compound(text):
    sia = SentimentIntensityAnalyzer()
    ss = sia.polarity_scores(text)
    return ss['compound']


def sia_attribution(all_sent):
    all_sent['sia_score'] = all_sent['sentence'].apply(sia_compound)
# all_sent.head()
    return all_sent

# applying textblob sentiment analysis


def detect_polarity(text):
    return TextBlob(text).sentiment


# for sentence in sentences:
#    print(sentence)
#    print(detect_polarity(sentence))

# applying to all sentences


def detect_polarity2(text):
    return TextBlob(text).sentiment.polarity

def detect_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def tb_polarity(all_sent):
    all_sent['tb_score'] = all_sent['sentence'].apply(detect_polarity2)
    all_sent['tb_subjectivity'] = all_sent['sentence'].apply(detect_subjectivity)

   # print(all_sent.head())

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    return all_sent


def calculate_average(line_lemm):
    features = dict()
    wordcount = 0
    pos_polarity = 0
    neg_polarity = 0
    a = 0
    b = 0
    z = 0
    sia_scores = []
    compound_scores_sia = list()
    positive_scores_sia = list()
    negative_scores_sia = list()
    polarity_score = []
    pos_average = 0
    neg_average = 0
    sia = SentimentIntensityAnalyzer()
    for line in line_lemm:
        compound_scores_sia.append(sia.polarity_scores(line)["compound"])
        positive_scores_sia.append(sia.polarity_scores(line)["pos"])
        negative_scores_sia.append(sia.polarity_scores(line)["neg"])
        Spolarity = TextBlob(line)
        polarity_score.append(Spolarity.sentiment.polarity)
        sia_scores.append(sia.polarity_scores(line))

    # Adding 1 to the final compound score to always have positive numbers
    # since some classifiers you'll use later don't work with negative numbers.

    features["mean_compound_sia"] = sum(compound_scores_sia) / len(compound_scores_sia) + 1
    features["mean_positive_sia"] = sum(positive_scores_sia) / len(positive_scores_sia)
    features["mean_negative_sia"] = sum(negative_scores_sia) / len(negative_scores_sia)

    return features

def graphdraw(all_sent):
    line_scores = all_sent.groupby('line').mean()[['sia_score',
                                                   'tb_score']].reset_index()
    first10 = line_scores.iloc[0:100]
    last10 = line_scores.iloc[-100:]

    # sns.relplot(kind='scatter', data=page_scores, x='page', y='tb_score')

    plt.scatter(x=line_scores['line'], y=line_scores['sia_score'], c='r')
    plt.scatter(x=line_scores['line'], y=line_scores['tb_score'], c='b')
    plt.show()
    return 0


central_function("/Users/kingsize/Desktop/Subs/TheDarkKnightLegends.srt")
central_function("/Users/kingsize/Desktop/Subs/TheDescentLegendas.srt")
central_function("/Users/kingsize/Desktop/Subs/UpLegendas.srt")
central_function("/Users/kingsize/Desktop/Subs/BackToSchoolLegendas.srt")
central_function("/Users/kingsize/Desktop/Subs/airplane 1980 -English.srt")
central_function("/Users/kingsize/Desktop/Subs/August Rush-English.srt")
central_function("/Users/kingsize/Desktop/Subs/Black.Swan.2010.720p.BluRay.x264.[YTS.MX]-English.srt")
central_function("/Users/kingsize/Desktop/Subs/Hot.Shots.1991.720p.BluRay.x264.AAC- YTS.MX-English.srt")
central_function("/Users/kingsize/Desktop/Subs/House.Of.Flying.Daggers.2004.1080p.BluRay-CLASSiC.en.srt")
central_function("/Users/kingsize/Desktop/Subs/love actually-English.srt")
central_function("/Users/kingsize/Desktop/Subs/Mr..Beans.Holiday.2007.1080p.720p.BluRay.x264. YTS.MX-English.srt")
central_function("/Users/kingsize/Desktop/Subs/My.Bodyguard.1980.720p.BluRay.x264.[YTS.MX]-English.srt")
central_function("/Users/kingsize/Desktop/Subs/my girl -English.srt")
central_function("/Users/kingsize/Desktop/Subs/Pink Flamingos.en.srt")
central_function("/Users/kingsize/Desktop/Subs/The Exorcist-English.srt")
central_function("/Users/kingsize/Desktop/Subs/prestige-English.srt")
central_function("/Users/kingsize/Desktop/Subs/The.Thin.Red.Line.srt")
central_function("/Users/kingsize/Desktop/Subs/When Harry Met Sally 1989 BRRip XviD AC3 VLiS-English.srt")
central_function("/Users/kingsize/Desktop/Subs/Silent Hill-English.srt")



# graphing results

def graphdraw(all_sent):
    line_scores = all_sent.groupby('line').mean()[['sia_score',
                                                   'tb_score']].reset_index()
    first10 = line_scores.iloc[0:100]
    last10 = line_scores.iloc[-100:]

    # sns.relplot(kind='scatter', data=page_scores, x='page', y='tb_score')

    plt.scatter(x=line_scores['line'], y=line_scores['sia_score'], c='r')
    plt.scatter(x=line_scores['line'], y=line_scores['tb_score'], c='b')
    plt.show()
    return 0
# plt.scatter(x=first10['page'], y=first10['sid_score'], c='r')
# plt.scatter(x=first10['page'], y=first10['tb_score'], c='b')

# plt.scatter(x=last10['page'], y=last10['sid_score'], c='r')
# plt.scatter(x=last10['page'], y=last10['tb_score'], c='b')
'''

features = dict()
wordcount = 0
compound_scores = list()
positive_scores = list()
negative_scores = list()
sia = SentimentIntensityAnalyzer()
for line in line_lemm:
        compound_scores.append(sia.polarity_scores(line)["compound"])
        positive_scores.append(sia.polarity_scores(line)["pos"])
        negative_scores.append(sia.polarity_scores(line)["neg"])

# Adding 1 to the final compound score to always have positive numbers
# since some classifiers you'll use later don't work with negative numbers.

features["mean_compound"] = sum(compound_scores) / len(compound_scores) + 1
features["mean_positive"] = sum(positive_scores) / len(positive_scores)
features["mean_negative"] = sum(negative_scores) / len(negative_scores)

print(features)'''

