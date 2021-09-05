import pandas as pd
import numpy as np

# text preprocessing
from nltk.tokenize import word_tokenize
import re

# preparing input to our model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# keras layers
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

import urllib.request
import zipfile
import os

import time


class text_to_emotion:
    def clean_text(self,data_clean_text):
        # remove hashtags and @usernames
        data_clean_text = re.sub(r"(#[\d\w\.]+)", '', data_clean_text)
        data_clean_text = re.sub(r"(@[\d\w\.]+)", '', data_clean_text)
        
        # tekenization using nltk
        data_clean_text = word_tokenize(data_clean_text)
        
        return data_clean_text

    def create_embedding_matrix(self,filepath, word_index, embedding_dim):
        vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        with open(filepath , encoding='utf-8') as f:
            for line in f:
                word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word] 
                    embedding_matrix[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]
        return embedding_matrix
    
    def train(self):
        # Number of labels: joy, anger, fear, sadness, neutral
        print("1")
        global num_classes
        num_classes = 5

        # Number of dimensions for word embedding
        global embed_num_dims
        embed_num_dims = 300

        # Max input length (max number of words) 
        global max_seq_len
        max_seq_len = 500

        global class_names
        class_names = ['Joy', 'Fear', 'Anger', 'Sadness', 'Neutral']

        global data_train
        data_train = pd.read_csv('static/data_text_to_emotion/data_train.csv', encoding='utf-8')
        global data_test
        data_test = pd.read_csv('static/data_text_to_emotion/data_test.csv', encoding='utf-8')

        global X_train
        X_train = data_train.Text
        global X_test
        X_test = data_test.Text

        global y_train
        y_train = data_train.Emotion
        global y_test
        y_test = data_test.Emotion

        global data
        data = data_train.append(data_test, ignore_index=True)

        global texts, texts_train, texts_test
        texts = [' '.join(self.clean_text(text)) for text in data.Text]

        texts_train = [' '.join(self.clean_text(text)) for text in X_train]
        texts_test = [' '.join(self.clean_text(text)) for text in X_test]

        global tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)

        global sequence_train, sequence_test
        sequence_train = tokenizer.texts_to_sequences(texts_train)
        sequence_test = tokenizer.texts_to_sequences(texts_test)

        global index_of_words
        index_of_words = tokenizer.word_index

        # vacab size is number of unique words + reserved 0 index for padding
        global vocab_size
        vocab_size = len(index_of_words) + 1

        # print('Number of unique words: {}'.format(len(index_of_words)))
        print("2")
        global X_train_pad, X_test_pad
        X_train_pad = pad_sequences(sequence_train, maxlen = max_seq_len )
        X_test_pad = pad_sequences(sequence_test, maxlen = max_seq_len )

        # X_train_pad

        global encoding
        encoding = {
            'joy': 0,
            'fear': 1,
            'anger': 2,
            'sadness': 3,
            'neutral': 4
        }

        # Integer labels
        y_train = [encoding[x] for x in data_train.Emotion]
        y_test = [encoding[x] for x in data_test.Emotion]
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # y_train
        print("3")
        global fname
        fname = 'static/data_text_to_emotion/wiki-news-300d-1M.vec'

        # if not os.path.isfile(fname):
        #     print('Downloading word vectors...')
        #     urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
        #                             'wiki-news-300d-1M.vec.zip')
        #     print('Unzipping...')
        #     with zipfile.ZipFile('wiki-news-300d-1M.vec.zip', 'r') as zip_ref:
        #         zip_ref.extractall('embeddings')
        #     print('done.')
            
        #     os.remove('wiki-news-300d-1M.vec.zip')

        global embedd_matrix
        embedd_matrix = self.create_embedding_matrix(fname, index_of_words, embed_num_dims)
        # embedd_matrix.shape

        # Inspect unseen words
        print("4")
        global new_words
        new_words = 0

        for word in index_of_words:
            entry = embedd_matrix[index_of_words[word]]
            if all(v == 0 for v in entry):
                new_words = new_words + 1

        # print('Words found in wiki vocab: ' + str(len(index_of_words) - new_words))
        # print('New words found: ' + str(new_words))

        # Embedding layer before the actaul BLSTM 
        global embedd_layer
        embedd_layer = Embedding(vocab_size,
                                embed_num_dims,
                                input_length = max_seq_len,
                                weights = [embedd_matrix],
                                trainable=False)

        # Convolution
        global kernel_size
        kernel_size = 3
        global filters
        filters = 256

        print("5")
        
        global model
        model = Sequential()
        model.add(embedd_layer)
        model.add(Conv1D(filters, kernel_size, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        # model.summary()

        # train model
        global batch_size
        batch_size = 256
        global epochs
        epochs = 6

        print("6")
        global hist
        hist = model.fit(X_train_pad, y_train, 
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test_pad,y_test))

        global predictions
        predictions = model.predict(X_test_pad)
        predictions = np.argmax(predictions, axis=1)
        predictions = [class_names[pred] for pred in predictions]

        # print("Accuracy: {:.2f}%".format(accuracy_score(data_test.Emotion, predictions) * 100))
        # print("\nF1 Score: {:.2f}".format(f1_score(data_test.Emotion, predictions, average='micro') * 100))
        print("training complete")

    def test(self,msg):
        message = [msg]

        seq = tokenizer.texts_to_sequences(message)
        padded = pad_sequences(seq, maxlen=max_seq_len)

        start_time = time.time()
        pred = model.predict(padded)

        print('Message: ' + str(message))
        print('predicted: {} ({:.2f} seconds)'.format(class_names[np.argmax(pred)], (time.time() - start_time)))
        return class_names[np.argmax(pred)]

# tte = text_to_emotion()
# tte.train()
# tte.test('Hello')
# tte.test('I love you')