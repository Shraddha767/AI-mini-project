# AI-mini-project
In this project I have implemeneted python program that predicts the price of stocks using a machine learning technique called Long Short-Term Memory (LSTM). Long short-term memory (LSTM) is an articial recurrent neural network (RNN) architecture used in the eld of deep learning. Unlike standard feed forward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video).LSTMs are widely used for sequence prediction problems and have proven to be extremely effective.The reason they work so well is because LSTM is able to store past information that is important, and forget the information that is not.
!pip install nbconvert
Requirement already satisfied: nbconvert in /usr/local/lib/python3.6/dist-packages (5.6.1)
Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.6/dist-packages (from nbconvert) (4.3.3)
Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.6/dist-packages (from nbconvert) (0.8.4)
Requirement already satisfied: defusedxml in /usr/local/lib/python3.6/dist-packages (from nbconvert) (0.6.0)
Requirement already satisfied: testpath in /usr/local/lib/python3.6/dist-packages (from nbconvert) (0.4.4)
Requirement already satisfied: bleach in /usr/local/lib/python3.6/dist-packages (from nbconvert) (3.2.1)
Requirement already satisfied: jinja2>=2.4 in /usr/local/lib/python3.6/dist-packages (from nbconvert) (2.11.2)
Requirement already satisfied: pygments in /usr/local/lib/python3.6/dist-packages (from nbconvert) (2.6.1)
Requirement already satisfied: jupyter-core in /usr/local/lib/python3.6/dist-packages (from nbconvert) (4.7.0)
Requirement already satisfied: entrypoints>=0.2.2 in /usr/local/lib/python3.6/dist-packages (from nbconvert) (0.3)
Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.6/dist-packages (from nbconvert) (1.4.3)
Requirement already satisfied: nbformat>=4.4 in /usr/local/lib/python3.6/dist-packages (from nbconvert) (5.0.8)
Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from traitlets>=4.2->nbconvert) (1.15.0)
Requirement already satisfied: decorator in /usr/local/lib/python3.6/dist-packages (from traitlets>=4.2->nbconvert)
(4. Requirement already satisfied: ipython-genutils in /usr/local/lib/python3.6/dist-packages (from
traitlets>=4.2->nbconve Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from
bleach->nbconvert) (20.4) Requirement already satisfied: webencodings in /usr/local/lib/python3.6/dist-packages (from
bleach->nbconvert) (0.5.1) Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.6/dist-packages
(from jinja2>=2.4->nbconvert) Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in
/usr/local/lib/python3.6/dist-packages (from nbformat>=4.4->n Requirement already satisfied: pyparsing>=2.0.2 in
/usr/local/lib/python3.6/dist-packages (from packaging->bleach->nbco
!jupyter nbconvert --to html AI_miniproject1.ipynb
[NbConvertApp] WARNING | pattern u'AI_miniproject1.ipynb' matched no files
This application is used to convert notebook files (*.ipynb) to various other
formats.
WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
Options
-------
Arguments that take values are actually convenience aliases to full
Configurables, whose aliases are listed on the help line. For more information
on full configurables, see '--help-all'.
--execute
Execute the notebook prior to export.
--allow-errors
Continue notebook execution even if one of the cells throws an error and include the error message in the cell ou
--no-input
Exclude input cells and output prompts from converted document.
This mode is ideal for generating code-free reports.
--stdout
Write notebook output to stdout instead of files.
--stdin
read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
--inplace
Run nbconvert in place, overwriting the existing notebook (only
relevant when converting to notebook format)
-y
Answer yes to any questions instead of prompting.
--clear-output
Clear output of current file and save in place,
overwriting the existing notebook.
--debug
set log level to logging.DEBUG (maximize logging output)
--no-prompt
Exclude input and output prompts from converted document.
--generate-config
generate default config file
--nbformat=<Enum> (NotebookExporter.nbformat_version)
Default: 4
Choices: [1, 2, 3, 4]
The nbformat version to write. Use this to downgrade notebooks.
--output-dir=<Unicode> (FilesWriter.build_directory)
Default: ''
Directory to write output(s) to. Defaults to output to the directory of each
notebook. To recover previous default behaviour (outputting to the current
working directory) use . as the flag value.
--writer=<DottedObjectName> (NbConvertApp.writer_class)
Default: 'FilesWriter'
Writer class used to write the results of the conversion
--log-level=<Enum> (Application.log_level)
Default: 30
Choices: (0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL')
Set the log level by value or name.
--reveal-prefix=<Unicode> (SlidesExporter.reveal_url_prefix)
Default: u''
The URL prefix for reveal.js (version 3.x). This defaults to the reveal CDN,
but can be any url pointing to a copy of reveal.js.
#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from google.colab import files
uploaded = files.upload()
Choose Files stock_data.csv
stock_data.csv(application/vnd.ms-excel) - 1185839 bytes, last modified: 5/25/2020 - 100% done
Saving stock_data.csv to stock_data (2).csv
#read the file
df=pd.read_csv("stock_data.csv")
#print the head
df.head()
Date Open High Low Close Volume OpenInt Stock
0 1984-09-07 0.42388 0.42902 0.41874 0.42388 23220030 0 AAPL
1 1984-09-10 0.42388 0.42516 0.41366 0.42134 18022532 0 AAPL
2 1984-09-11 0.42516 0.43668 0.42516 0.42902 42498199 0 AAPL
3 1984-09-12 0.42902 0.43157 0.41618 0.41618 37125801 0 AAPL
4 1984-09-13 0.43927 0.44052 0.43927 0.43927 57822062 0 AAPL
#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()
#Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])
#Converting the dataframe to a numpy array
dataset = data.values
#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8)
#Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1)) scaled_data
= scaler.fit_transform(dataset)
#Create the scaled training data set
train_data = scaled_data[0:training_data_len , : ]
#Split the data into x_train and y_train data sets
x_train=[]
y_train = []
for i in range(60,len(train_data)):
x_train.append(train_data[i-60:i,0])
y_train.append(train_data[i,0])
#Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
#Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)
15609/15609 [==============================] - 361s 23ms/step - loss: 5.1502e-04
<tensorflow.python.keras.callbacks.History at 0x7f504b887f28>
#Test data set
test_data = scaled_data[training_data_len - 60: , : ]
#Create the x_test and y_test data sets
x_test = []
y_test = dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this
for i in range(60,len(test_data)):
x_test.append(test_data[i-60:i,0])
#Convert x_test to a numpy array
x_test = np.array(x_test)
#Reshape the data into the shape accepted by the LSTM x_test =
np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
#Getting the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)#Undo scaling
#Calculate/Get the value of RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse
4.028245627902073
#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
#Show the valid and predicted prices
valid
