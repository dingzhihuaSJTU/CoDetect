import numpy as np

def covxy(x:np.ndarray, y:np.ndarray)->float:
    """
    ## calculate the covariance
    Args:
        Input: 
            x, y: the series to calculate the covariance
    """
    covxy = np.mean((x - x.mean()) * (y - y.mean()))
    return covxy

def relate(x:np.ndarray, y:np.ndarray)->float:
    """
    ## calculate the correlation coefficient

    Args:
        Input: 
            x, y: the series to calculate the correlation coefficient
    """
    cov = covxy(x, y)
    std_x = np.std(x)+1e-6
    std_y = np.std(y)+1e-6
    return cov/(std_x*std_y)

def DCA(x: np.ndarray, windowsize: int, cov: bool = False)->np.ndarray:
    """
    ## Do the DCA for series

    Args:
        Input: 
            x: the series to calculate the DCS
            windowsize: the size of sliding windows
            cov: default for False. If True, output is the covariance series. 
        Output:
            Dynamic correlate series
    """
    n = len(x)
    x = np.array(x)
    corr_list = []
    for i in range(n-2*windowsize):
        x1 = x[i: i+windowsize]
        x2 = x[i+windowsize: i+2*windowsize]
        if not cov:
            corr_list.append(relate(x1, x2))
        else:
            # If True, output is the covariance series. 
            corr_list.append(covxy(x1, x2))
    return np.array(corr_list)

def scalenon(x: float)-> float:
    return (1-np.exp(-x))/(1+np.exp(-x))

def DCV(data:np.ndarray, windowsize:int)->float:
    """
    ## Measure the non-stationarity of series

    Args:
        Input: 
            data: the series to calculate the DCV;
            windowsize: the size of sliding windows
    """
    corr_np = DCA(data, windowsize, False)
    return scalenon(np.abs(np.std(corr_np)/np.mean(corr_np)+1e-6))