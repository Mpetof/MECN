import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

class MemoryDNN:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):

        self.net = net
        self.training_interval = training_interval      # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1

        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

    def _build_net(self):
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]),
                nn.ReLU(),
                nn.Linear(self.net[1], self.net[2]),
                nn.ReLU(),
                nn.Linear(self.net[2], self.net[3]),
                nn.Sigmoid()
        )

    def remember(self, h, m):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        # encoding the entry
        self.remember(h, m)
        # train the DNN every multiple steps

        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])


        # train the DNN
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001) 
        criterion = nn.BCELoss()
        self.model.train()
        optimizer.zero_grad()
        predict = self.model(h_train)
        loss = criterion(predict, m_train)
        loss.backward()
        optimizer.step()

        self.cost = loss.item()
        assert(self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, h, k = 1, mode = 'OP'):
        # to have batch dimension when feed into Tensor
        h = torch.Tensor(h[np.newaxis, :])

        self.model.eval()
        m_pred = self.model(h)
        m_pred = m_pred.detach().numpy()


        if mode == 'OP':
            return self.knm(m_pred[0], k)
        elif mode == 'KNN':
            return self.knn(m_pred[0], k)
        elif mode == 'OPN':
            return self.opn(m_pred[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN' or 'OPN'")

    def knm(self, m, k = 1):
        # return k order-preserving binary actions
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        m_list.append(1*(m>0.5))

        if k > 1:
            # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
            m_abs = abs(m-0.5)
            idx_list = np.argsort(m_abs)[:k-1]
            for i in range(k-1):
                if m[idx_list[i]] >0.5:
                    # set the \hat{x}_{t,(k-1)} to 0
                    m_list.append(1*(m - m[idx_list[i]] > 0))
                else:
                    # set the \hat{x}_{t,(k-1)} to 1
                    m_list.append(1*(m - m[idx_list[i]] >= 0))

        return m_list
    
    def opn(self, m, k= 1):
        return self.knm(m,k)+self.knm(m+np.random.normal(0,1,len(m)),k)

    def knn(self, m, k = 1):
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) == 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]


    def plot_cost(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.style.use('seaborn')
        fig, ax = plt.subplots(figsize=(15,8))
    
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.savefig("loss.png", dpi=300)
        plt.show()
        
class Model:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval = 10, 
        batch_size = 10,  
        memory_size = 1000,
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
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]),
                nn.ReLU(),
                nn.Linear(self.net[1], self.net[2]),
                nn.ReLU(),
                nn.Linear(self.net[2], self.net[3]),
                nn.Sigmoid()
        )
        
    def decode (self, h, K) :       
        h = torch.Tensor(h[np.newaxis, :]) # fit DNN dimension
        
        self.model.eval()
        x_pred = self.model (h)
        x_pred = x_pred.detach().numpy()
        
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
        
        h_train = torch.Tensor(batch_memory [:, 0 : self.net[0]])
        x_train = torch.Tensor(batch_memory [:, self.net[0] : ])
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001)
        criterion = nn.BCELoss()
        self.model.train()
        optimizer.zero_grad()
        
        predict = self.model(h_train)
        loss = criterion(predict, x_train)
        loss.backward()
        optimizer.step()

        self.cost = loss.item()
        assert(self.cost > 0)
        self.loss.append(self.cost)
        
    def plot_cost(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.style.use('seaborn')
        fig, ax = plt.subplots(figsize=(15,8))
    
        plt.plot(np.arange(len(self.loss))*self.mem_train_interval, self.loss)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.savefig("loss.png", dpi=300)
        plt.show()