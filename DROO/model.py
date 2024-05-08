import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Model:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval = 10, 
        batch_size = 10,  
        memory_size = 1000,
        output_graph = False
    ):
        self.net = net
        self.lr = learning_rate
        
        self.mem_size = memory_size
        self.mem_counter = 0
        self.mem = np.zeros(( self.mem_size , self.net[0] + self.net[-1]))
        self.batch_size = batch_size
        self.mem_train_interval = training_interval
        
        self.loss = []
        
        self.build_model()
        
    def build_model (self):
        self.model = keras.Sequential([
                    layers.Dense(self.net[1], activation='relu'),  # the first hidden layer
                    layers.Dense(self.net[2], activation='relu'),  # the second hidden layer
                    layers.Dense(self.net[-1], activation='sigmoid')  # the output layer
                ])
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])
        
    def decode (self, h, K) :
        h = h[np.newaxis, :] # fit DNN dimension
        x_pred = self.model.predict (h)
        return self.quanti (x_pred [0], K) # quantization step
    
    def quanti (self, x, K):
        x_list = [] 
        x_list.append (1 * (x > 0.5)) # # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        
        if K > 1:
            # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
            x_abs = abs(x-0.5)
            idx_list = np.argsort(x_abs)[:K-1]

            for i in range(K-1):
                if x[idx_list[i]] >0.5:
                    # set the \hat{x}_{t,(k-1)} to 0
                    x_list.append(1*(x - x[idx_list[i]] > 0))
                else:
                    # set the \hat{x}_{t,(k-1)} to 1
                    x_list.append(1*(x - x[idx_list[i]] >= 0))
        
        return x_list
    
    def encode (self, h, x):
        self.memory (h, x)
        if self.mem_counter % self.mem_train_interval == 0:
            self.train ()
        
    def memory (self, h, x):
        idx = self.mem_counter % self.mem_size
        self.mem[idx, :] = np.hstack((h, x))
        self.mem_counter += 1
    
    def train (self):
        if self.mem_counter > self.mem_size:
            sample_index = np.random.choice(self.mem_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.mem_counter, size=self.batch_size)
        batch_memory = self.mem[sample_index, :]
        
        h_train = batch_memory [:, 0 : self.net[0]]
        x_train = batch_memory [:, self.net[0] : ]
        
        fitting = self.model.fit (h_train, x_train, verbose=0)
        
        loss = fitting.history ["loss"][0]
        assert(loss > 0)
        self.loss.append(loss)
        