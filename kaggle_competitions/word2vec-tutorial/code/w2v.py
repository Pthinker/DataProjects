# coding=utf-8

import pandas as pd
from bs4 import BeautifulSoup
import re
import os
from nltk.corpus import stopwords
import nltk.data
import logging
from gensim.models import word2vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import time


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def review_to_wordlist(raw_review, remove_stopwords=False):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z0-9]", " ", review_text)
    words = letters_only.lower().split()
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return words


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))

    return sentences

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        if counter%1000 == 0:
            print "Review %d of %d" % (counter, len(reviews))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


def averageVector(model, num_features):
    print "Creating average feature vecs for train reviews"
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review.decode("utf8"), remove_stopwords=True))
    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

    print "Creating average feature vecs for test reviews"
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review.decode("utf8"), remove_stopwords=True))
    testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

    forest = RandomForestClassifier(n_estimators=100)

    print "Fitting a random forest to labeled training data..."
    forest = forest.fit(trainDataVecs, train["sentiment"])

    result = forest.predict(testDataVecs)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("../result/Word2Vec_AverageVectors.csv", index=False, quoting=3)


def create_bag_of_centroids(wordlist, word_centroid_map):
    # The number of clusters is equal to the highest cluster index in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1
    
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1

    return bag_of_centroids


def clustering(model, train, test):
    print "Clustering..."
    start = time.time() # Start time

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an average of 5 words per cluster
    word_vectors = model.wv.syn0
    num_clusters = word_vectors.shape[0] / 5

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)

    end = time.time()
    elapsed = end - start
    print "Time taken for K Means clustering: ", elapsed, "seconds."

    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number                                                                                            
    word_centroid_map = dict(zip(model.wv.index2word, idx))

    train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")

    # Transform the training set reviews into bags of centroids
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review.decode("utf8"), remove_stopwords=True))
    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review.decode("utf8"), remove_stopwords=True))
    test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")
    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    forest = RandomForestClassifier(n_estimators=100)
    print "Fitting a random forest to labeled training data..."
    forest = forest.fit(train_centroids, train["sentiment"])
    result = forest.predict(test_centroids)

    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("../result/BagOfCentroids.csv", index=False, quoting=3)


def getModel(sentences, num_features=300):
    # Set values for various parameters
    min_word_count = 40   # Minimum word count                        
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size                                                                                    
    downsampling = 1e-3   # Downsample setting for frequent words

    print "Training model..."
    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, 
            window=context, sample=downsampling)
    model.init_sims(replace=True)
    model_name = "../model/300features_40minwords_10context"
    model.save(model_name)

    return model


if __name__ == "__main__":
    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    train = pd.read_csv("../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("../data/testData.tsv", header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv("../data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    print "Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews\n" % (
            train["review"].size, test["review"].size, unlabeled_train["review"].size)

    # get sentences
    sentences = []
    print "Parsing sentences from training set"
    for review in train["review"]:
        sentences += review_to_sentences(review.decode("utf8"), tokenizer)
    print "Parsing sentences from unlabeled set"
    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review.decode("utf8"), tokenizer)

    num_features = 300    # Word vector dimensionality
    model = getModel(sentences, num_features)

    #averageVector(model, num_features)
    clustering(model, train, test)
    


