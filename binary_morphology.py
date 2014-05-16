import numpy as np
from matplotlib.pyplot import figure, ion

default_struct = np.array([0,1,0,
                           1,1,1,
                           0,1,0]).reshape(3,3)

def invert(A):
    """ Returns the *inverse* of the input array. """
    arr = np.ones_like(A)
    arr[np.where(A > 0)] = 0
    return arr

def union(A, B):
    """
    Return the *union* of the input arrays.

    Raises:
        ValueError: The input arrays do not have the same shape.

    """
    if A.shape != B.shape:
        raise ValueError('Arrays must be the same shape.')
    arr = np.copy(A)
    arr[np.where(B > 0)] = 1
    return arr

def intersection(A, B):
    """
    Return the *intersection* of the input arrays.

    Raises:
        ValueError: The input arrays do not have the same shape.

    """
    if A.shape != B.shape:
        raise ValueError('Arrays must be the same shape.')
    arr = np.copy(A)
    arr[np.where(B == 0)] = 0
    return arr

def equals(A, B):
    """
    Returns True if the arrays are equal, False otherwise.

    Raises:
        ValueError: The input arrays do not have the same shape.

    """
    if A.shape != B.shape:
        raise ValueError('Arrays must be the same shape.')
    return np.all(A == B)

def complement(A, B):
    """ Returns the *relative complement* of B in A. """
    return intersection(A, invert(B))

def difference(A, B):
    """ Returns the *symmetric difference* of A and B. """
    return union(complement(A,B), complement(B, A))

def pad(A, val = 0, width = 1):
    """
    Pad the input array. e.g.

        >>> default_struct
        array([[0, 1, 0]
               [1, 1, 1]
               [0, 1, 0]])
        >>> pad(default_struct, val = 0, width = 2)
        array([[0, 0, 0, 0, 0, 0, 0]
               [0, 0, 0, 0, 0, 0, 0]
               [0, 0, 0, 1, 0, 0, 0]
               [0, 0, 1, 1, 1, 0, 0]
               [0, 0, 0, 1, 0, 0, 0]
               [0, 0, 0, 0, 0, 0, 0]
               [0, 0, 0, 0, 0, 0, 0]])
    Args:
        * A (numpy.ndarray): The input array.
        * val (int): (Optional) The value used to pad the array.  0 is used if 
          no value is provided.
        * width (int): (Optional) The number of rows and columns of padding to 
          add.  1 row and column is added if no width is provided.

    """
    col = np.zeros((A.shape[0], width), dtype = np.bool)
    row = np.zeros((width, A.shape[1] + width*2), dtype = np.bool)

    if val:
        col.fill(val)
        row.fill(val)

    arr = np.column_stack((col, A, col))
    return np.vstack((row, arr, row))

def dilate(A, struct = default_struct):
    """
    Apply morphological *dilation* to the input array.

    Args:
        * A (numpy.ndarray): The input array.
        * struct (np.mdarray): (Optional) The morphological structuring element.
          Currently only 3x3 structs are supported.  If no struct is provided, 
          the following default will be used::

            [[0, 1, 0]
             [1, 1, 1]
             [0, 1, 0]]

    Raises:
        * ValueError: The shape of the structuring element is not 3x3.

    """
    if not (struct.shape[0] == 3 and struct.shape[1] == 3):
        raise ValueError('Currently only supports 3x3 structuring elements.')

    A = pad(A)
    arr = np.copy(A)

    for row in range(A.shape[0]-2):
        for col in range(A.shape[1]-2):
            if A[row+1, col+1]:
                arr[row:row+3, col:col+3] = union(arr[row:row+3, col:col+3], struct)

    return arr[1:-1, 1:-1]

def erode(A, struct = default_struct): 
    """
    Apply morphological *erosion* to the input array.

    Args:
        * A (numpy.ndarray): The input array.
        * struct (np.mdarray): (Optional) The morphological structuring element.
          Currently only 3x3 structs are supported.  If no struct is provided, 
          the following default will be used::

            [[0, 1, 0]
             [1, 1, 1]
             [0, 1, 0]]

    Raises:
        * ValueError: The shape of the structuring element is not 3x3.

    """
    if not (struct.shape[0] == 3 and struct.shape[1] == 3):
        raise ValueError('Currently only supports 3x3 structuring elements.')

    arr = np.zeros_like(A)
    A = pad(A)

    for row in range(A.shape[0]-2):
        for col in range(A.shape[1]-2):
            arr[row,col] = equals(struct, intersection(A[row:row+3, col:col+3], struct))

    return arr


def open(A, struct = default_struct):
    """
    Apply morphological *opening* to the input array.

    Args:
        * A (numpy.ndarray): The input array.
        * struct (np.mdarray): (Optional) The morphological structuring element.
          Currently only 3x3 structs are supported.  If no struct is provided, 
          the following default will be used::

            [[0, 1, 0]
             [1, 1, 1]
             [0, 1, 0]]

    """
    return dilate(erode(A, struct), struct)

def close(A, struct = default_struct):
    """
    Apply morphological *closing* to the input array.

    Args:
        * A (numpy.ndarray): The input array.
        * struct (np.mdarray): (Optional) The morphological structuring element.
          Currently only 3x3 structs are supported.  If no struct is provided, 
          the following default will be used::

            [[0, 1, 0]
             [1, 1, 1]
             [0, 1, 0]]

    """
    return erode(dilate(A, struct), struct)

def outline(A):
    """ Returns the outline of the input array. """
    struct = np.ones((3,3))
    return complement(dilate(A, struct), A)

def stretch(A, scale):
    """
    Stretch the input array.

    FIXME: Does this live in this library ?

    Args:
        * A (numpy.ndarray): The input array.
        * scale (tuple): The amount the stretch in (x, y).

    """
    xscale, yscale = scale
    return np.repeat(np.repeat(A, yscale, axis=0), xscale, axis = 1)

def fill_holes(A, connectivity = 8):
    """ Returns a copy of A with all holes filled in. """
    prev = np.zeros_like(A)
    prev = pad(prev, 1)
    mask = pad(invert(A), 1)

    if connectivity == 8:
      struct = default_struct
    elif connectivity == 4:
      struct = np.ones((3,3))
    else:
      raise ValueError("'connectivity' must be either 8 or 4.")
    
    while 1:
        next = intersection(dilate(prev, struct),  mask)
        if equals(next, prev):
            break
        prev = next

    return invert(next[1:-1, 1:-1])

def display(array, title = None):
    fig = figure()
    ax = fig.add_subplot(111)

    if title is not None:
        ax.set_title(title)

    ax.imshow(array, interpolation = 'none')