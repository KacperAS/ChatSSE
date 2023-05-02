import numpy as np #for maths
import json #for json files
import re #for regular expressions
import tensorflow as tf #for machine learning
import random #for the randomizer
import spacy #for NLP
import tkinter as tk
import nltk #for NLP
import glob
import tkinter as tk
from tkinter import messagebox, font as tkFont
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# import data for training
data_file_path = glob.glob("**/data.json", recursive=True)
if not data_file_path:
    raise FileNotFoundError("Could not locate 'data.json' file")

with open(r"C:\Users\kackr\OneDrive\Pulpit\data.json") as f:
    intents = json.load(f)

# preprocess data
def preprocessing(line):
    # convert to lowercase
    line = line.lower()
    
    # remove punctuation and digits
    line = re.sub(r'[^\w\s]|\d', '', line)
    
    # remove stop words
    line = " ".join([word for word in line.split() if word not in stop_words])
    
    # lemmatize words
    line = " ".join([lemmatizer.lemmatize(word) for word in line.split()])
    
    return line

# get text and intent title from json data
inputs, targets = [], []
classes = []
intent_doc = {}

for intent in intents['intents']:
    if intent['intent'] not in classes:
        classes.append(intent['intent'])
    if intent['intent'] not in intent_doc:
        intent_doc[intent['intent']] = []
        
    for text in intent['text']:
        inputs.append(preprocessing(text))
        targets.append(intent['intent'])
        
    for response in intent['responses']:
        intent_doc[intent['intent']].append(response)

# Tokenize input data
def tokenize_data(input_list):
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, max_features=500)
    input_seq = vectorizer.fit_transform(input_list).toarray()
    return vectorizer, input_seq

vectorizer, input_tensor = tokenize_data(inputs)

def create_categorical_target(targets):
    word={}
    categorical_target=[]
    counter=0
    for trg in targets:
        if trg not in word:
            word[trg]=counter
            counter+=1
        categorical_target.append(word[trg])
    
    categorical_tensor = tf.keras.utils.to_categorical(categorical_target, num_classes=len(word), dtype='int32')
    return categorical_tensor, dict((v,k) for k, v in word.items())

# preprocess output data
target_tensor, trg_index_word = create_categorical_target(targets)

print('input shape: {} and output shape: {}'.format(input_tensor.shape, target_tensor.shape))

# hyperparameters
epochs=50
vocab_size = input_tensor.shape[1]
embed_dim=512
units=128
target_length=target_tensor.shape[1]

# build MLP Model with tensorflow
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(embed_dim, input_shape=(vocab_size,), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(target_length, activation='softmax')
])

initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

# train the model
model.fit(input_tensor, target_tensor, epochs=epochs, callbacks=[early_stop])

def response(sentence):
    # Preprocess input sentence using TfidfVectorizer
    sent_seq = vectorizer.transform([preprocessing(sentence)]).toarray()

    # predict the category of input sentences
    pred = model(sent_seq)

    pred_class = np.argmax(pred.numpy(), axis=1)
    
    # choose a random response for predicted sentence
    return random.choice(intent_doc[trg_index_word[pred_class[0]]]), trg_index_word[pred_class[0]]

# chat from terminal
# print("Note: Enter 'quit' to break the loop.")
# while True:
#    input_ = input('You: ')
#    if input_.lower() == 'quit':
#        break
#    res, typ = response(input_)
#    print('Bot: {} -- TYPE: {}'.format(res, typ))
#    print()

# GUI Functions
class ChatWindow:
    def __init__(self, canvas):
        self.canvas = canvas
        self.message_height = 0

    def create_text_bubble(self, message, side):
        margin = 10
        padding = 5

        text_id = self.canvas.create_text(margin if side == "left" else self.canvas.winfo_width() - margin,
                                          self.message_height + margin,
                                          text=message,
                                          anchor=tk.NW if side == "left" else tk.NE,
                                          width=self.canvas.winfo_width() - 2 * margin,
                                          font=text_font, tags=side)

        bbox = self.canvas.bbox(text_id)

        rect_id = self.canvas.create_rectangle(bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding,
                                               fill="#2ecc71" if side == "right" else "#3498db", outline="")
        self.canvas.tag_lower(rect_id, text_id)

        self.message_height += bbox[3] - bbox[1] + 2 * padding + margin
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def send_message(self, message, side):
        self.create_text_bubble(message, side)
        self.canvas.yview_moveto(1.0)

def send():
    message = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", tk.END)

    if message != '':
        chat_window.send_message(message, "right")
        res, typ = response(message)
        chat_window.send_message("ChatSSE: " + res + " -- TYPE: " + typ, "left")

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        base.destroy()

# GUI Setup
base = tk.Tk()
base.title("Chatbot")
base.geometry("500x650")
base.configure(bg="#2C3E50")
base.resizable(width=False, height=False)

# Custom fonts
title_font = tkFont.Font(family="Helvetica", size=18, weight="bold")
text_font = tkFont.Font(family="Helvetica", size=12)

# Create Chat Window
ChatLog = tk.Canvas(base, bd=0, bg="#F5F5F5", height="8", width="50")
ChatLog.configure(highlightbackground="#7F8C8D", highlightthickness=3, relief="groove")

ChatLog.config(state=tk.DISABLED)

# Bind scrollbar to Chat Window
scrollbar = tk.Scrollbar(base, command=ChatLog.yview, cursor="heart")
scrollbar.configure(bg="#7F8C8D", troughcolor="#F5F5F5", highlightbackground="#7F8C8D", highlightthickness=3)
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = tk.Button(base, font=("Helvetica", 16, 'bold'), text="Send", width="10", height=2,
                    bd=0, bg="#3498DB", activebackground="#2980B9", fg='#ffffff',
                    command=send)
SendButton.configure(highlightbackground="#7F8C8D", highlightthickness=3, relief="groove")

SendButton.place(x=25, y=580, height=50)
ChatLog.config(scrollregion=ChatLog.bbox("all"))

# Create the box to enter message
EntryBox = tk.Text(base, bd=0, bg="#F5F5F5", width="29", height="5", font=text_font)
EntryBox.configure(highlightbackground="#7F8C8D", highlightthickness=3, relief="groove")

# Place all components on the screen
scrollbar.place(x=455, y=80, height=356)
ChatLog.place(x=25, y=80, height=356, width=424)
EntryBox.place(x=25, y=450, height=120, width=424)
SendButton.place(x=25, y=580, height=50)

chat_window = ChatWindow(ChatLog)

base.protocol("WM_DELETE_WINDOW", on_closing)

# Add a title label
title_label = tk.Label(base, text="ChatSSE - Student Enterprise Chatbot", bg="#2C3E50", fg="#ECF0F1", font=title_font)
title_label.place(x=25, y=20)

# Run the GUI window
base.mainloop()