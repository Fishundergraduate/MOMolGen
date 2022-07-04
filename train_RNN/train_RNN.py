import csv
import datetime
#import itertools
#import operator
import numpy as np
#import nltk
import os
#import sys
#from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation,TimeDistributed
from keras.layers import LSTM,GRU
#from keras.layers.embeddings import Embedding
from keras.layers import Embedding
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers import Dropout
import numpy as np
#import random
#import sys
from keras.utils.np_utils import to_categorical
#from keras.preprocessing import sequence
from keras.models import model_from_json, Model
from keras import Input
from keras.callbacks import TensorBoard, Callback
#from make_smile import ziprocess_organic,process_zinc_data
from make_smile import zinc_data_with_bracket_original,zinc_processed_with_bracket

from keras.layers import Conv1D, MaxPooling1D
#from combine_bond_atom import organic, process_organic,bond_atom
from keras.utils import pad_sequences

from tensorflow import distribute, data
import tensorflow as tf


def load_data():

    sen_space=[]
    f = open(os.path.dirname(__file__)+'/../data/smile_trainning.csv', 'rb')
    reader = csv.reader(f)
    for row in reader:
        #word_space[row].append(reader[row])
        #print word_sapce
        sen_space.append(row)
    #print sen_space
    f.close()

    element_table=["Cu","Ti","Zr","Ga","Ge","As","Se","Br","Si","Zn","Cl","Be","Ca","Na","Sr","Ir","Li","Rb","Cs","Fr","Be","Mg",
            "Ca","Sr","Ba","Ra","Sc","La","Ac","Ti","Zr","Nb","Ta","Db","Cr","Mo","Sg","Mn","Tc","Re","Bh","Fe","Ru","Os","Hs","Co","Rh",
            "Ir","Mt","Ni","Pd","Pt","Ds","Cu","Ag","Au","Rg","Zn","Cd","Hg","Cn","Al","Ga","In","Tl","Nh","Si","Ge","Sn","Pb","Fl",
            "As","Sb","Bi","Mc","Se","Te","Po","Lv","Cl","Br","At","Ts","He","Ne","Ar","Kr","Xe","Rn","Og"]
    #print sen_space
    word1=sen_space[0]
    word_space=list(word1[0])
    end="\n"
    #start="st"
    #word_space.insert(0,end)
    word_space.append(end)
    #print word_space
    #print len(sen_space)
    all_smile=[]
    #print word_space
    #for i in range(len(all_smile)):

    for i in range(len(sen_space)):
        word1=sen_space[i]
        word_space=list(word1[0])
        word=[]
        #word_space.insert(0,end)
        j=0
        while j<len(word_space):
            word_space1=[]
            word_space1.append(word_space[j])
            if j+1<len(word_space):
                word_space1.append(word_space[j+1])
                word_space2=''.join(word_space1)
            else:
                word_space1.insert(0,word_space[j-1])
                word_space2=''.join(word_space1)
            if word_space2 not in element_table:
                word.append(word_space[j])
                j=j+1
            else:
                word.append(word_space2)
                j=j+2

        word.append(end)
        all_smile.append(list(word))
    #print all_smile
    val=[]
    for i in range(len(all_smile)):
        for j in range(len(all_smile[i])):
            if all_smile[i][j] not in val:
                val.append(all_smile[i][j])
    #print val
    val.remove("\n")
    val.insert(0,"\n")

    return val, all_smile


""" def organic_data():
    sen_space=[]
    #f = open('/Users/yang/smiles.csv', 'rb')
    f = open('/Users/yang/LSTM-chemical-project/make_sm.csv', 'rb')
    #f = open('/Users/yang/LSTM-chemical-project/smile_trainning.csv', 'rb')
    reader = csv.reader(f)
    for row in reader:
        #word_space[row].append(reader[row])
        #print word_sapce
        sen_space.append(row)
    #print sen_space
    f.close()

    element_table=["Cu","Ti","Zr","Ga","Ge","As","Se","Br","Si","Zn","Cl","Be","Ca","Na","Sr","Ir","Li","Rb","Cs","Fr","Be","Mg",
            "Ca","Sr","Ba","Ra","Sc","La","Ac","Ti","Zr","Nb","Ta","Db","Cr","Mo","Sg","Mn","Tc","Re","Bh","Fe","Ru","Os","Hs","Co","Rh",
            "Ir","Mt","Ni","Pd","Pt","Ds","Cu","Ag","Au","Rg","Zn","Cd","Hg","Cn","Al","Ga","In","Tl","Nh","Si","Ge","Sn","Pb","Fl",
            "As","Sb","Bi","Mc","Se","Te","Po","Lv","Cl","Br","At","Ts","He","Ne","Ar","Kr","Xe","Rn","Og"]
    #print sen_space
    word1=sen_space[0]
    word_space=list(word1[0])
    end="\n"
    #start="st"
    #word_space.insert(0,end)
    word_space.append(end)
    #print word_space
    #print len(sen_space)
    all_smile=[]
    #print word_space
    #for i in range(len(all_smile)):

    for i in range(len(sen_space)):
        word1=sen_space[i]
        word_space=list(word1[0])
        word=[]
        #word_space.insert(0,end)
        j=0
        while j<len(word_space):
            word_space1=[]
            word_space1.append(word_space[j])
            if j+1<len(word_space):
                word_space1.append(word_space[j+1])
                word_space2=''.join(word_space1)
            else:
                word_space1.insert(0,word_space[j-1])
                word_space2=''.join(word_space1)
            if word_space2 not in element_table:
                word.append(word_space[j])
                j=j+1
            else:
                word.append(word_space2)
                j=j+2

        word.append(end)
        all_smile.append(list(word))
    #print all_smile
    val=[]
    for i in range(len(all_smile)):
        for j in range(len(all_smile[i])):
            if all_smile[i][j] not in val:
                val.append(all_smile[i][j])
    #print val
    val.remove("\n")
    val.insert(0,"\n")

    return val, all_smile
 """

def prepare_data(smiles,all_smile):
    all_smile_index=[]
    for i in range(len(all_smile)):
        smile_index=[]
        for j in range(len(all_smile[i])):
            smile_index.append(smiles.index(all_smile[i][j]))
        all_smile_index.append(smile_index)
    X_train=all_smile_index
    y_train=[]
    for i in range(len(X_train)):

        x1=X_train[i]
        x2=x1[1:len(x1)]
        x2.append(0)
        y_train.append(x2)

    return X_train,y_train

"""
def generate_smile(model,val):
    end="\n"
    start_smile_index= [val.index("C")]
    new_smile=[]

    while not start_smile_index[-1] == val.index(end):
        predictions=model.predict(start_smile_index)
        ##next atom probability
        smf=[]
        for i in range (len(X)):
            sm=[]
            for j in range(len(X[i])):
                #if np.argmax(predictions[i][j])=!0
                sm.append(np.argmax(predictions[i][j]))
            smf.append(sm)

        #print sm
        #print smf
        #print len(sm)

        new_smile.append(sampled_word)
    #sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    #return new_sentence
"""


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def createModel(vocab_size: int, embed_size: int, N: int):
        input = Input(shape=(N,))
        x = Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=N, mask_zero=True)(input)
        x = GRU(units=256,activation='tanh',return_sequences=True)(x)
        #x = LSTM(output_dim=256, input_shape=(81,64),activation='tanh',return_sequences=True)(x)
        x = Dropout(.2)(x)
        x = GRU(units=256,activation='tanh',return_sequences=True)(x)
        #x = LSTM(output_dim=256, input_shape=(81,64),activation='tanh',return_sequences=True)(x)
        x = Dropout(.2)(x)
        x = TimeDistributed(Dense(embed_size, activation='softmax'))(x)
        model = Model(inputs=input, outputs=x)

        """ model.add(Dropout(0.2))
        model.add(GRU(units=256,activation='tanh',return_sequences=True, input_shape=(None , )))
        #model.add(LSTM(output_dim=1000, activation='sigmoid',return_sequences=True))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(embed_size, activation='softmax')))"""
        optimizer=Adam(learning_rate=0.01) 
        
        #print(model.summary())
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

class EarlyStoppingByTimer(Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """
    def __init__(self, patience=0, startTime=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=+9),'JST')), timeLimit=datetime.timedelta(hours=23)):
        super(EarlyStoppingByTimer, self).__init__()
        self._time = startTime
        self._timeLimit = timeLimit
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self._recentTrainBegin = datetime.timedelta()
        self._JST = datetime.timezone(datetime.timedelta(hours=+9),'JST')

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_begin(self, epoch, logs=None):
        self._recentTrainBegin = datetime.datetime.now(self._JST)
        return super().on_epoch_begin(epoch, logs)
        
    def on_epoch_end(self, epoch, logs=None):
        _now = datetime.datetime.now(self._JST)
        _recentTimeDelta = _now - self._recentTrainBegin

        if _now - self._time >= self._timeLimit + _recentTimeDelta:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Saving Models in This JOB")
                self.model.set_weights(self.best_weights)
                self.on_train_end()
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()


    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

if __name__ == "__main__":
    startTime = datetime.datetime.now()
    dataOpt = data.Options()
    dataOpt.experimental_distribute.auto_shard_policy = data.experimental.AutoShardPolicy.DATA
    
    #comOpt = distribute.experimental.CommunicationOptions(implementation=distribute.experimental.CommunicationImplementation.NCCL)
    #strategy = distribute.MultiWorkerMirroredStrategy(communication_options=comOpt)
    strategy = distribute.MirroredStrategy()
    smile=zinc_data_with_bracket_original()
    valcabulary,all_smile=zinc_processed_with_bracket(smile)
    # print(valcabulary)
    # print(len(all_smile))
    X_train,y_train=prepare_data(valcabulary,all_smile)
  
    maxlen=81

    X = pad_sequences(X_train, maxlen=81, dtype='int32',
        padding='post', truncating='pre', value=0.)
    y = pad_sequences(y_train, maxlen=81, dtype='int32',
        padding='post', truncating='pre', value=0.)
    #X= sequence.pad_sequences(X_train, maxlen=81, dtype='int32',
    #    padding='post', truncating='pre', value=0.)
    #y = sequence.pad_sequences(y_train, maxlen=81, dtype='int32',
    #    padding='post', truncating='pre', value=0.)
    
    y_train_one_hot = np.array([to_categorical(sent_label, num_classes=len(valcabulary)) for sent_label in y])
    # print (y_train_one_hot.shape)
    vocab_size=len(valcabulary)
    embed_size=len(valcabulary)

    
    N=X.shape[1]


    with strategy.scope():
        """ model = Sequential()

        model.add(Embedding(input_dim=vocab_size, output_dim=len(valcabulary), input_length=N,mask_zero=False, input_shape=(None, vocab_size)))
        model.add(GRU(units=256,activation='tanh',return_sequences=True))
        #model.add(LSTM(output_dim=256, input_shape=(81,64),activation='tanh',return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(units=256,activation='tanh',return_sequences=True))
        #model.add(LSTM(output_dim=1000, activation='sigmoid',return_sequences=True))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(embed_size, activation='softmax')))
        optimizer=Adam(lr=0.01)
        print(model.summary())
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) """
        model = createModel(vocab_size=vocab_size,embed_size=embed_size,N=N)
    X_nd_train, X_nd_valid = X[:int(len(X)*0.9)], X[int(len(X)*0.9):]
    y_nd_train_one_hot, y_nd_valid_one_hot = y_train_one_hot[:int(len(y_train_one_hot)*0.9)], y_train_one_hot[int(len(y_train_one_hot)*0.9):]
    #print(X.shape,X_nd_train.shape,y_train_one_hot.shape,y_nd_train_one_hot.shape)
    #print(X.dtype)
    
    trainDataset = data.Dataset.zip((data.Dataset.from_tensor_slices(X_nd_train), data.Dataset.from_tensor_slices(tf.cast(y_nd_train_one_hot, dtype=tf.float32)))).shuffle(buffer_size=N).batch(512 * strategy.num_replicas_in_sync).prefetch(data.experimental.AUTOTUNE).with_options(dataOpt)
    validDataset = data.Dataset.zip((data.Dataset.from_tensor_slices(X_nd_valid), data.Dataset.from_tensor_slices(tf.cast(y_nd_valid_one_hot, dtype=tf.float32))))                       .batch(512 * strategy.num_replicas_in_sync).prefetch(data.experimental.AUTOTUNE).with_options(dataOpt)
    #traindDataset = strategy.experimental_distribute_dataset(trainDataset)
    #validDataset = strategy.experimental_distribute_dataset(validDataset)
    #print(trainDataset.element_spec)
    """ print(X_nd_train[1])
    print(y_nd_train_one_hot[1])
    for elem in trainDataset.as_numpy_iterator():
        print(elem)
        break
    print(tf.convert_to_tensor(X_nd_train,dtype=tf.int32)[1]) """
    #TODO: Dataset with validation
    
    tensorboard_callback = TensorBoard(log_dir="../tensorboard_logs", profile_batch=5)
    earlystoppingByTimer = EarlyStoppingByTimer(timeLimit=datetime.timedelta(hours=2))
    #model.fit(x=trainDataset,validation_data=validDataset,epochs=100, batch_size=512, callbacks=[tensorboard_callback,earlystoppingByTimer])
    #model.fit(x=X,y=y_train_one_hot,epochs=100, batch_size=512, validation_split=.1, callbacks=[tensorboard_callback,earlystoppingByTimer])
    #model.fit(x=trainDataset,epochs=100, validation_data=validDataset, callbacks=[tensorboard_callback,earlystoppingByTimer])
    model.fit(x=trainDataset,epochs=100, validation_data=validDataset, callbacks=[earlystoppingByTimer])
    save_model(model)
