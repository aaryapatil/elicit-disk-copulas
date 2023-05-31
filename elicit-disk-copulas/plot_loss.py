import pandas as pd


def moving_average(x, n):
    
    y = np.zeros(len(x))
    y_err = np.zeros(len(x))
    y[0] = np.nan
    y_err[0] = np.nan
    
    for i in range(1,len(x)):
        
        if i < n:
            y[i] = np.mean(x[:i])
            y_err[i] = np.std(x[:i])
        else:
            y[i] = np.mean(x[i-n:i])
            y_err[i] = np.std(x[i-n:i])
            
    return y, y_err

def plot(x, label, show_band=True):

    mv, mv_err = moving_average(x, 100)

    if show_band:
        plt.fill_between(np.arange(len(mv)), mv-mv_err, mv+mv_err, alpha=0.4)
    plt.plot(mv, label=label, linewidth=1, color='r') 
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Iteration number', fontsize=16)
    plt.yscale('symlog')
    plt.yticks([-2, -1.5, -1, -0.5])
    plt.savefig('loss.pdf', format='pdf', bbox_inches = 'tight')


# Read the training loss
loss = pd.read_csv('loss').values
# Plot loss
plot(loss, 'loss')