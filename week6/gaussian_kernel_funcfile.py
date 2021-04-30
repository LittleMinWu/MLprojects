import numpy as np

def gaussian_kernel0(distances):
    weights=np.exp(-0*(distances**2))
    return weights/np.sum(weights)
def gaussian_kernel1(distances):
    weights=np.exp(-1*(distances**2))
    return weights/np.sum(weights)
def gaussian_kernel5(distances):
    weights=np.exp(-5*(distances**2))
    return weights/np.sum(weights)
def gaussian_kernel10(distances):
    weights=np.exp(-10*(distances**2))
    return weights/np.sum(weights)
def gaussian_kernel25(distances):
    weights=np.exp(-25*(distances**2))
    return weights/np.sum(weights)
def gaussian_kernel50(distances):
    weights=np.exp(-50*(distances**2))
    return weights/np.sum(weights)
def gaussian_kernel100(distances):
    weights=np.exp(-100*(distances**2))
    return weights/np.sum(weights)
def gaussian_kernel200(distances):
    weights=np.exp(-200*(distances**2))
    return weights/np.sum(weights)