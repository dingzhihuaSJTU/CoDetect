
import numpy as np
from NonStationarity import DCA
import scipy.stats as stats
import scipy.integrate as integrate

# ----------------------------------------------------------------
# Correlation-Based Change Point Detection with Sliding Windows
# ----------------------------------------------------------------

def ConfidenceLevel(tau: int, 
                    snr: float, 
                    l1: float = -1, 
                    l2: float = 1, 
                    )->tuple[float, float]:
    """
    Args:
        Input:
            l1, l2: control the threshold; 
            tau: the sliding windows size;
            snr: SNR; 

        Output:
            rho_s: threshold ; 
            1-\alpha: confidence level
    
    """
    A = snr + 1/np.sqrt(tau) * l1
    B = snr+1 + np.sqrt(2/(tau-1)) * l2

    def integrand(z):
        y = snr+1 + z * np.sqrt(2/(tau-1))
        threshold = A / B * y
        return (1 - stats.norm.cdf((threshold - snr) / (1/np.sqrt(tau)))) * stats.norm.pdf(z)

    result, error = integrate.quad(integrand, -np.inf, np.inf)
    return A/B, round(float(result), 4)

def CoDetect(data: np.ndarray, 
             tau: int, 
             rho_s: float, 
             min_windows: int = None, 
             )->list[int]:
    """
    Args:
        Input:
            data: the series(n-dimension);
            tau: the sliding windows size;
            rho_s: the threshold; 
            min_windows: the minimum windows for detecting change point
                         (one change point in the windows at most);

        Output:
            change point index; 
    """
    dcs = DCA(data, tau)
    if min_windows is None:
        min_windows = 4*tau
    i = 0
    ans = []
    while i < len(dcs):
        if dcs[i] <= rho_s:
            # print(i)
            p_i = _codetect(dcs, tau, min_windows, i)
            ans.append(p_i)
            # i+=1
            i = p_i+min_windows-2*tau
        else:
            i+=1
    ans.append(len(data))
    return ans

def _codetect(data: np.ndarray, 
              tau: int, 
              lm: int, 
              index: int, 
              )->int:
    """
    Args:
        data: dcs; 
        tau: the sliding windows size;
        lm: the minimum windows for detecting change point
            (one change point in the windows at most);
        index: the begin index for segment; 
        return: p_i
    
    """
    
    
    sum_dcs_2tau_dict = {}
    for i in range(index, index+lm+1-2*tau):
        sum_dcs_2tau_dict[i] = np.sum(data[i: i+2*tau])
    min_key = min(sum_dcs_2tau_dict, key=sum_dcs_2tau_dict.get)
    return min_key+2*tau
 
if __name__ == "__main__":
    data = np.load("./Datasets/SyntheticDatasets/length_1000_peroid_16_snr_50.0_wave_sine_change_mean_changerange_2.0.npy")
    rhos, confile = ConfidenceLevel(16, 50, -1, 1)
    print(rhos, confile)
    codetect_index = CoDetect(data[:500], 16, rhos)

    from Tools import Mask2Index, PltSettings
    import matplotlib.pyplot as plt
    from CraftedData import CraftedChangePointMask

    mask = CraftedChangePointMask(1000, 16)
    PltSettings()
    plt.plot(data[:500])
    index = Mask2Index(mask[:500])
    for i in index:
        plt.axvline(i)
    for i in codetect_index:
        plt.axvline(i, color="r")
    plt.show()



# def CoDetect(data, tau, jump=None, threshold=None):
#     data = standardize(data)
#     length = len(data)
#     ans = []
#     if jump is None:
#         jump = int(tau/2)
#     # print(jump)
#     i_list = [i for i in range(2*tau)]
#     std_ei = {}
#     for index in range(0, length-4*tau, 1):
#         Ei = np.zeros((2*tau, ))
#         for i in range(2*tau):
#             x0 = data[index+int(i_list[i]): index+tau+int(i_list[i])].reshape(-1, 1)
#             xt = data[index+tau+int(i_list[i]): index+2*tau+int(i_list[i])].reshape(-1, 1)
#             x0_mean = np.mean(x0, axis=0)
#             xt_mean = np.mean(xt, axis=0)
#             std_x = np.std(x0)+1e-6
#             std_y = np.std(xt)+1e-6
#             Ei[i] = np.mean((x0-x0_mean)*(xt-xt_mean))/(std_x*std_y)
#         std_ei[index] = np.mean(Ei)
    
    
#     min_size = 4*tau
    
#     if threshold is None:
#         threshold = np.mean(list(std_ei.values()))

#     filtered_dict = {k: v for k, v in std_ei.items() if v < threshold}

#     while filtered_dict != {}:
#         max_key = min(filtered_dict, key=filtered_dict.get)
#         ans.append(max_key+2*tau)
#         filtered_dict = {k: v for k, v in filtered_dict.items() if k > max_key + min_size or k < max_key - min_size}

#     ans.append(length-1)
#     print('CorrWindow: {}'.format(len(ans)))
#     # plt.plot(list(std_ei.values()))
#     # plt.axhline(threshold)
#     # for i in ans:
#     #     plt.axvline(i)
#     # plt.show()
#     return ans