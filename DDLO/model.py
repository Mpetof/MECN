import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Memory:
    def __init__(self, size, input_dim, output_dim):
        self.size = size
        self.mem_counter = 0
        self.memory = np.zeros((self.size, input_dim + output_dim))
    
    def store(self, h, x):
        idx = self.mem_counter % self.size
        self.memory[idx, :] = np.hstack((h, x))
        self.mem_counter += 1

    def sample(self, batch_size):
        if self.mem_counter > self.size:
            sample_index = np.random.choice(self.size, size=batch_size)
        else:
            sample_index = np.random.choice(self.mem_counter, size=batch_size)
        return self.memory[sample_index, :]
    
class Model:
    def __init__(
        self,
        net,
        memory,
        learning_rate = 0.01,
        training_interval = 10, 
        batch_size = 10,  
        output_graph = False
    ):
        self.net = net
        self.lr = learning_rate
        
        self.memory = memory
        self.batch_size = batch_size
        self.mem_train_interval = training_interval
        
        self.loss = []
        
        self.build_model()
        
    def build_model (self):
        with tf.device('/CPU:0'):
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
        self.memory.store (h, x)
        if self.memory.mem_counter % self.mem_train_interval == 0:
            self.train()
    
    def train (self):
        batch_memory = self.memory.sample(self.batch_size)
        
        h_train = batch_memory [:, 0 : self.net[0]]
        x_train = batch_memory [:, self.net[0] : ]
        
        fitting = self.model.fit (h_train, x_train, verbose=0)
        
        loss = fitting.history ["loss"][0]
        assert(loss > 0)
        self.loss.append(loss)
    
    def plot_loss(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.style.use('seaborn')
        fig, ax = plt.subplots(figsize=(15,8))
    
        plt.plot(np.arange(len(self.loss))*self.mem_train_interval, self.loss)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.savefig("loss.png", dpi=300)
        plt.show()
        