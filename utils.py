import torch
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def binarise_matrix(M, thresh_val, thresh_type='pos'):
    """
    Returns a binarised matrix using a threshold.

    Parameters
    ----------
    M : numpy.ndarray
        The matrix to binarise
    thresh_val : float
        The cut-off value
    thresh_type : str (default 'pos'):
        'pos' : values less than thresh_val are given 0. Other values are given 1.
        'neg' : values greater than thresh_val are given 0. Other values are given 1.
        'abs' : absolute values less than the absolute value of thresh_val are given 0. Other values are given 1.

    Returns
    -------
    out : numpy.ndarray
        The binarised matrix.

    Raises
    ------
    ValueError
        When thresh_type is not 'pos', 'neg' or 'abs'.

    Examples
    --------
    >>> M = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
    >>> print(binarise_matrix(M, 2))
    [0 0 0 0 0 0 0 1 1 1 1]
    >>> print(binarise_matrix(M, -2, 'neg'))
    [1 1 1 1 0 0 0 0 0 0 0]
    >>> print(binarise_matrix(M, 2, 'neg'))
    [1 1 1 1 1 1 1 1 0 0 0]
    >>> print(binarise_matrix(M, 2, 'abs'))
    [1 1 1 1 0 0 0 1 1 1 1]
    """

    # Define yes and no conditions
    if thresh_type == 'pos':
        y = M >= thresh_val
        n = M < thresh_val
    elif thresh_type == 'neg':
        y = M <= thresh_val
        n = M > thresh_val
    elif thresh_type == 'abs':
        y = abs(M) >= abs(thresh_val)
        n = abs(M) < abs(thresh_val)
    else:
        raise ValueError("Input parameter thresh_type needs to be 'pos', 'neg' or 'abs'.")

    # Binarise using conditions
    M_bin = M.copy()
    M_bin[y] = 1
    M_bin[n] = 0

    return M_bin

def set_rand_seed(seed: int) -> None:
    """
    Sets the random seed for the random, NumPy and PyTorch libaries

    Parameters
    ----------
    seed : int
        Random seed to use
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_cv_test_split(M, train_pct, cv_pct):
    """
    Randomly splits an array of indices into three sets

    Parameters
    ----------
    M : numpy.ndarray
        Indices to split
    train_pct : float
        Percentage of indices for the training set
    cv_pct : float
        Percentage of indices for the cross validation set

    Returns
    -------
    out : tuple
        3 numpy arrays for the training, cross validation and test sets
    """

    np.random.shuffle(M)
    len_M = len(M)
    test_pct = 1 - train_pct - cv_pct
    test_count = int(test_pct * len_M)
    cv_count = int(cv_pct * len_M)
    cv_end = len_M - test_count
    train_end = cv_end - cv_count
    return M[:train_end], M[train_end:cv_end], M[cv_end:]

def upper_indices(n):
    """
    Returns the indices of the upper triangle of a matrix to use in loops

    Parameters
    ----------
    n : numpy.ndarray
        Number of rows in the matrix

    Returns
    -------
    out : zip
        Indices for the upper triangle of the matrix

    Examples
    --------
    >>> for i, j in upper_indices(3): print(i,j)
    0 1
    0 2
    1 2
    """

    return zip(*np.triu_indices(n, 1))

def validate_binary(*M):
    """
    Raises an exception if matrices contain elements other than 1 or 0.

    Parameters
    ----------
    M : numpy.ndarray
        A sequence of matrices

    Raises
    ------
    ValueError
        When at least one of the matrices contains elements other than 1 or 0
    """

    for m in M:
        if ((m != 0) & (m != 1)).any():
            raise ValueError('An input matrix is not a binary matrix')

def validate_loopless(*M):
    """
    Raises an exception if matrices contain elements other than 0 along the diagonal.

    Parameters
    ----------
    M : numpy.ndarray
        A sequence of matrices

    Raises
    ------
    ValueError
        When at least one of the matrices contains elements other than 0 along the diagonal
    """

    for m in M:
        if (m.diagonal() != 0).any():
            raise ValueError('An input matrix contains an element other than 0 along the diagonal')

def validate_method(*M):
    """
    Raises an exception if strings are not valid weighted method types

    Parameters
    ----------
    M : str
        A sequence of strings

    Raises
    ------
    ValueError
        When at least one of the strings is not a valid weighted method type
    """

    for m in M:
        if m not in ['max', 'sum']:
            raise ValueError('A weighted method type is invalid')

def validate_square(*M):
    """
    Raises an exception if matrices are not square.

    Parameters
    ----------
    M : numpy.ndarray
        A sequence of matrices

    Raises
    ------
    ValueError
        When at least one of the matrices is not square
    """

    for m in M:
        n = len(m)
        for len_ax in m.shape:
            if len_ax != n:
                raise ValueError('An input matrix is not square')

def validate_thresh_type(*M):
    """
    Raises an exception if strings are not valid threshold types

    Parameters
    ----------
    M : str
        A sequence of strings

    Raises
    ------
    ValueError
        When at least one of the strings is not a valid threshold type
    """

    for m in M:
        if m not in ['pos', 'neg', 'abs']:
            raise ValueError('A threshold type is invalid')

def validate_symmetric(*M):
    """
    Raises an exception if matrices are not symmetric

    Parameters
    ----------
    M : numpy.ndarray
        A sequence of matrices

    Raises
    ------
    ValueError
        When at least one of the matrices is not symmetric
    """

    for m in M:
        len_m = len(m)
        if (abs(m - m.T)[np.triu_indices(len_m, 1)] > 1e-5).any():
            raise ValueError('An input matrix is not symmetric')