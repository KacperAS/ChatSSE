import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only

# Import the dataset
import pandas as pd
data = pd.read_csv("path/to/your/dataset.csv")

# Extract the questions and answers from the dataset
questions = data['questions'].tolist()
answers = data['answers'].tolist()

# Update the corpus with the questions and answers from the dataset
corpus = []
for i in range(len(questions)):
    corpus.append(questions[i])
    corpus.append(answers[i])

def LemTokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Define a function to generate responses
def generateResponse(userInput):
    # Define the corpus and the tf-idf vectorizer
    corpus = ["How can I view my class schedule?",
              "How do I add or drop a class?",
              "How can I access my grades and transcripts?",
              "How can I pay my tuition?",
              "How can I apply for financial aid?",
              "How can I find a tutor?"]
    tfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = tfidfVec.fit_transform(corpus)

    # Use cosine similarity to generate a response
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf==0):
        return "I'm sorry, I don't understand your question. Can you please rephrase it or provide more context?"
    else:
        return corpus[idx]

# Define a function to handle user input
def chatbot():
    userInput = input("You: ")
    while userInput != 'bye':
        userInput = input("You: ")
        response = generateResponse(userInput)
        print("Chatbot: ", response)

# Start the chatbot
chatbot()