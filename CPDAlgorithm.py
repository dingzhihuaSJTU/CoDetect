
import numpy as np
import ruptures as rpt

# ----------------------------------------------------------------
# Traditional Algorithms
# ----------------------------------------------------------------

def Pelt(data: np.ndarray, windowsize: int, jump: int = 1) -> list:
    """
    ```Binary Segment```

    Args:
        data: Array
        minsize, jump
    Return:
        bkps: list of the change points
    """
    # data_pre = Data_Detect_Preprocess(data, window_size=windowsize)
    data_pre = data.reshape(-1, 1)
    n = data_pre.shape[0]
    dim = data_pre.shape[1]
    sigma = np.std(data_pre)
    # print(n, dim, sigma)

    algo = rpt.Pelt(model="l2", min_size=windowsize, jump=jump).fit(data_pre)
    my_bkps = algo.predict(pen=np.log(n) * dim * sigma**2)
    print("Pelt: {}".format(len(my_bkps)))
    return my_bkps

def BinSeg(data: np.ndarray, windowsize: int, jump: int = 1) -> list:
    """
    ```Binary Segment```

    Args:
        data: Array
        minsize, jump
    Return:
        bkps: list of the change points
    """
    # data_pre = Data_Detect_Preprocess(data, window_size=windowsize)
    data_pre = data.reshape(-1, 1)
    n = data_pre.shape[0]
    dim = data_pre.shape[1]
    sigma = np.std(data_pre)
    # print(n, dim, sigma)

    algo = rpt.Binseg(model="l2", min_size=windowsize, jump=jump).fit(data_pre)
    my_bkps = algo.predict(pen=np.log(n) * dim * sigma**2)
    print("BinSeg: {}".format(len(my_bkps)))
    return my_bkps

def BottomUp(data: np.ndarray, windowsize: int, jump: int = 1) -> list:
    """
    ```Binary Segment```

    Args:
        data: Array
        minsize, jump
    Return:
        bkps: list of the change points
    """
    # data_pre = Data_Detect_Preprocess(data, window_size=windowsize)
    data_pre = data.reshape(-1, 1)
    n = data_pre.shape[0]
    dim = data_pre.shape[1]
    sigma = np.std(data_pre)
    # print(n, dim, sigma)

    algo = rpt.BottomUp(model="l2", min_size=windowsize, jump=jump).fit(data_pre)
    my_bkps = algo.predict(pen=np.log(n) * dim * sigma**2)
    print("BottomUp: {}".format(len(my_bkps)))
    return my_bkps

def Window(data: np.ndarray, windowsize: int, jump: int = 1) -> list:
    """
    ```Binary Segment```

    Args:
        data: Array
        minsize, jump
    Return:
        bkps: list of the change points
    """
    # data_pre = Data_Detect_Preprocess(data, window_size=windowsize)
    data_pre = data.reshape(-1, 1)
    n = data_pre.shape[0]
    dim = data_pre.shape[1]
    sigma = np.std(data_pre)
    # print(n, dim, sigma)

    algo = rpt.Window(model="l2", min_size=windowsize, jump=jump).fit(data_pre)
    my_bkps = algo.predict(pen=np.log(n) * dim * sigma**2)
    print("Window: {}".format(len(my_bkps)))
    return my_bkps

