import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import numpy as np

def plot_rate (file, rolling_intv = 50):
    
    Q_rate = np.loadtxt(file)
    df = pd.DataFrame(Q_rate)
    
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15,8))
    
    plt.plot(np.arange(len(Q_rate))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(Q_rate))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color = 'b', alpha = 0.2)
    plt.ylabel('Normalized Computation Rate')
    plt.xlabel('Time Frames')
    plt.savefig('Q rate.png', dpi=300)
    plt.show()    

def plot_loss(file, rolling_intv = 50):
    
    loss = np.loadtxt(file)
    df = pd.DataFrame(loss)
    
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15,8))
    rolling_intv = 20

    plt.plot(np.arange(len(loss))+1, df.rolling(rolling_intv, min_periods=1).mean(), 'r')
    
    plt.ylabel('Loss')
    plt.xlabel('Time Frames')
    plt.savefig('loss.png', dpi=300)
    plt.show()
    
def main ():
    file = "Q_rate.txt"
    plot_rate (file)
    file_l = "loss.txt"
    plot_loss (file_l)
    
main ()