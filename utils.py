import numpy as np

def exclude_by_value(array, col, val, pct=100.):
    """Remove numpy array elements that are equals to a 
    specific value.
    
    Args:
        array: numpy array to be elements removed.
        col: integer represents the indis of the column 
            that will be searched.
        val: value in the column to be removed.
        pct: percentage of removed elements. 
            example: (float)100 removes all.
    """

    rows = np.where(array[:,col] == val)[0]
    rows_not = np.where(array[:,col] != val)[0]
    perm = np.random.permutation(rows.shape[0])
    pct_left = np.int(rows.shape[0] * (100 - pct) / 100)
    shuff_rows = rows[perm][0:pct_left]
    index = np.concatenate((rows_not, shuff_rows))

    return array[index]

