import numpy as np


def F1Score(trueindex: np.ndarray, 
            detectindex: np.ndarray, 
            threshold: int = 8, 
            ) -> float:
    TP = 0
    for index in trueindex:
        for _index in detectindex:
            if abs(index-_index) <= threshold:
                TP += 1
                break
    Prec = TP/len(detectindex)
    Rec = TP/len(trueindex)
    return 2*(Prec*Rec)/(Prec+Rec+1e-6)


def Mask2Index(changepointmask: np.ndarray):
    index = np.nonzero(np.diff(changepointmask))[0]+1
    index = np.concatenate((index, [changepointmask.shape[0]-1]))
    return index


# =============================================
# ============= Base Curve ====================
# =============================================

# The **Amplitude** is 1. 

def sine(length: int, peroid: int)->np.ndarray: 
    """
    args:
        input:
            length: the total length of curve; 
            peroid;
    """
    x = np.array([i for i in range(length)])
    y = np.sin(2*np.pi/peroid*x)
    return y

def square(length: int, peroid: int)->np.ndarray: 
    """
    args:
        input:
            length: the total length of curve; 
            peroid;
    """
    x = np.array([i for i in range(length)])
    y = np.sign(np.sin(2*np.pi/peroid*x))
    return y

def triangle(length: int, peroid: int)->np.ndarray: 
    """
    args:
        input:
            length: the total length of curve; 
            peroid;
    """
    x = np.array([i for i in range(length)])
    y = 4 * np.abs(np.mod(x, peroid) - peroid / 2) / peroid-1
    return y

# =============================================
# ================== End ======================
# =============================================

def PltSettings(picsize: tuple[int, int] = (12, 6)):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['font.size']=30
    rcParams['svg.fonttype']='none'
    plt.rc('font',family='Times New Roman')
    rcParams['mathtext.fontset']='stix'
    rcParams['axes.grid']=True
    rcParams['axes.axisbelow']=True
    rcParams['grid.linestyle']='--'
    rcParams['xtick.direction']='in'
    rcParams['ytick.direction']='in'
    rcParams['savefig.dpi'] = 600
    rcParams['figure.dpi'] = 600
    rcParams['figure.figsize']=picsize

