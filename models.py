from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from collections import deque
from keras.layers import Bidirectional
from keras.layers import BatchNormalization

class ResearchModels():
    def __init__(self, nb_classes, model, input_shape):

        # Set defaults.
        
        self.load_model = load_model
        #self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        #if self.nb_classes >= 10:
        #   metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        print("Loading LSTM model.")
        self.input_shape = input_shape
        self.model = self.lstm()
        # Now compile the network.
        optimizer = Adam(learning_rate=0.00005)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer,
                           metrics=metrics)
	
        print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        
    
        
        
        #LSTM layer
        
        model.add(LSTM(256, return_sequences=False, input_shape=self.input_shape, dropout=0.2))
        # Dropout layer
        model.add(Dropout(0.4))
        
        # Dense layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='relu'))
   

        # Output layer
        model.add(Dense(2, activation='sigmoid'))

        return model    
    


# def lstm(self):
#         """Build a simple LSTM network. We pass the extracted features from
#         our CNN to this model predomenently."""
#         # Model.
#         print (self.input_shape)
        
#         model = Sequential()
        
#         model.add(LSTM(128, return_sequences=False,
#                        input_shape=self.input_shape,
#                        dropout=0.2))
       
#         model.add(Flatten())
        
#         model.add(Dense(1024, activation='relu'))
#         model.add(Dense(512, activation='relu'))
#         model.add(Dropout(0.5))
#        # model.add(Dense(256, activation='relu'))
#         model.add(Dense(self.nb_classes, activation='sigmoid'))
    
