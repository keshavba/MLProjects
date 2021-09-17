import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import re
from nltk.corpus import stopwords
import joblib

STOPWORDS = set(stopwords.words('english'))
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

# Place limits on consumer complaints for LSTM modeling
MAX_NO_WORDS = 50000 # Maximum number of words to be used for modeling
MAX_SEQUENCE_LENGTH = 250 # Maximum number of words in each complaint
EMBEDDING_DIM = 100

# Display graph of number of complaints for each product
def display_complaints_graph(df):
    data = pd.Series.value_counts(self=df['product'], ascending=False)
    plt.figure()
    data.plot(kind='bar', title='Number of Complaints for each product')
    plt.show()

# Print consumer complaint associated with the given index in the pandas dataframe
def print_complaint(df, index):
    example = df[df.index == index][['consumer_complaint_narrative', 'product']].values[0]
    if len(example) > 0:
        print('\n')
        print(example[0])
        print('Product:', example[1])

# Clean the passed text string by replacing the mentioned symbols with spaces and blanks
def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def main():
    try:
        model = keras.models.load_model('lstm_model.h5')
    except:
        # Read consumer_complaints.csv using pandas and consolidate the data labels
        df = pd.read_csv('consumer_complaints.csv', low_memory=False)
        df = df.dropna(axis='index')
        df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].astype(str)
        df.loc[df['product'] == 'Credit reporting', 'product'] = 'Credit reporting, credit repair services, or other personal consumer reports'
        df.loc[df['product'] == 'Credit card', 'product'] = 'Credit card or prepaid card'
        df.loc[df['product'] == 'Payday loan', 'product'] = 'Payday loan, title loan, or personal loan'
        df.loc[df['product'] == 'Virtual currency', 'product'] = 'Money transfer, virtual currency, or money service'
        df = df[df['product'] != 'Other financial service']

        # Pre-process the text using the clean_text function for usage in the LSTM model
        df = df.reset_index(drop=True)
        df['consumer_complaint_narrative'] =  df['consumer_complaint_narrative'].apply(clean_text)
        df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].str.replace('\d+', ' ', regex=True)

        tokenizer = Tokenizer(num_words=MAX_NO_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(df['consumer_complaint_narrative'].values)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        # Truncate and pad the input sequences for modeling
        X = tokenizer.texts_to_sequences(df['consumer_complaint_narrative'].values)
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', X.shape)

        # Convert categorical labels into numbers
        Y = pd.get_dummies(df['product']).values
        print('Shape of label tensor:', Y.shape)

        # Train test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        print('Shape of train splits:', X_train.shape, Y_train.shape)
        print('Shape of test splits:', X_test.shape, Y_test.shape)
        joblib.dump(X_test, 'x_test')
        joblib.dump(Y_test, 'y_test')

        # Set up and train the LSTM model
        model = Sequential()
        model.add(Embedding(MAX_NO_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        epochs = 3
        batch_size = 2

        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        # Save the model to .h5 file
        model.save('lstm_model.h5')

        # Display a graph with the loss of the model on the train and test sets
        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        # Display a graph with the accuracy of the model on the train and test sets
        plt.title('Accuracy')
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.show()

    # Evaluate the model on the test set
    accuracy = model.evaluate(joblib.load('x_test'), joblib.load('y_test'))
    print('Test set\n Lost: {:0.3f}\n Accuracy: {:0.3f}'.format(accuracy[0], accuracy[1]))

    # Testing the model with a new complaint
    new_complaint = ['I am a victim of identity theft and someone stole my identity and personal information to open up a Visa credit card account with Bank of America. The following Bank of America Visa credit card account does not belong to me : XXXX.']
    print('We will test the model with the following new complaint: ', new_complaint)
    tokenizer = Tokenizer(num_words=MAX_NO_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    seq = tokenizer.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    labels = ['Credit reporting, credit repair services, or other personal consumer reports', 'Debt collection', 'Mortgage', 'Credit card or prepaid card', 'Student loan', 'Bank account or service', 'Checking or savings account', 'Consumer Loan', 'Payday loan, title loan, or personal loan', 'Vehicle loan or lease', 'Money transfer, virtual currency, or money service', 'Money transfers', 'Prepaid card']
    print("LSTM model's prediction on new complaint: ", pred, labels[np.argmax(pred)])

if __name__ == '__main__':
    main()