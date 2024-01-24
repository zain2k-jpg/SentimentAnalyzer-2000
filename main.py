import nltk
from nltk.corpus import movie_reviews
from random import shuffle

# Download the movie_reviews dataset
nltk.download('movie_reviews')
# Get the movie reviews and their labels (positive or negative)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents to mix positive and negative reviews
shuffle(documents)
# Define a feature extractor function (simple bag-of-words)
def document_features(document):
    words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in words)
    return features

# Get the 2000 most frequent words as features
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

# Extract features for each document
featuresets = [(document_features(d), c) for (d, c) in documents]
# Split the data into a training set and a testing set (80% training, 20% testing)
train_set, test_set = featuresets[:1600], featuresets[1600:]
# Train a Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)
# Evaluate the accuracy of the classifier
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f'Accuracy: {accuracy:.2%}')
# Example of making predictions
example_review = "I loved the movie. The plot was engaging, and the actors were fantastic!"
features = document_features(example_review.split())
prediction = classifier.classify(features)
print(f'Sentiment Prediction: {prediction}')
