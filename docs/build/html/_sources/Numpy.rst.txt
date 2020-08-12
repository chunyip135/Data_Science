.. _Numpy:

===============================
Numpy
===============================

* 
  Create a numpy array

  .. code-block:: python

       # Create an 1d array from a list
       import numpy as np
       list1 = [0,1,2,3,4]
       arr1d = np.array(list1)

       # Print the array and its type
       print(type(arr1d))
       arr1d

       #> class 'numpy.ndarray'
       #> array([0, 1, 2, 3, 4])


  * 
    if apply function, it is perform on every item in the array

    .. code-block:: python

         # Add 2 to each element of arr1d
         arr1d + 2

         #> array([2, 3, 4, 5, 6])


    * once numpy array is created, cannot increase its size

* 
  2D array

  .. code-block:: python

       # Create a 2d array from a list of lists
       list2 = [[0,1,2], [3,4,5], [6,7,8]]
       arr2d = np.array(list2)
       arr2d

       #> array([[0, 1, 2],
       #>        [3, 4, 5],
       #>        [6, 7, 8]])

* 
  specify datatype by setting the dtype argument


  * float, int, bool ,str,object
  * 
    control memory allocations


    * 
      float32 , float64, int8, int16 ..

      .. code-block:: python

           # Create a float 2d array
           arr2d_f = np.array(list2, dtype='float')
           arr2d_f

           #> array([[ 0.,  1.,  2.],
           #>        [ 3.,  4.,  5.],
           #>        [ 6.,  7.,  8.]])

           arr2d_f.dtype # print its data type

  * 
    convert to different dtype

    .. code-block:: python

         # Convert to 'int' datatype
         arr2d_f.astype('int')

         #> array([[0, 1, 2],
         #>        [3, 4, 5],
         #>        [6, 7, 8]])

         # Convert to int then to str datatype
         arr2d_f.astype('int').astype('str')

         #> array([['0', '1', '2'],
         #>        ['3', '4', '5'],
         #>        ['6', '7', '8']],
         #>       dtype='U21')

         # Create a boolean array
         arr2d_b = np.array([1, 0, 10], dtype='bool')
         arr2d_b

         #> array([ True, False,  True], dtype=bool)

         # Create an object array to hold numbers as well as strings
         arr1d_obj = np.array([1, 'a'], dtype='object')
         arr1d_obj

         #> array([1, 'a'], dtype=object)

* 
  convert array to list

  .. code-block:: python

       arr1d_obj.tolist()

* 
  Basic syntax

  .. code-block:: python

       arr1.ndim # dimension
       arr1.shape # shape ( how may rows and columns)
       arr1.size # total number of elements
       arr1.dtype

       # Extract the first 2 rows and columns
       arr2[:2, :2]

       arr2[arr2 > 1]

       # Reverse only the row positions
       arr2[::-1, ]

       # Reverse only column
       arr2[:, ::-1]

       # reverse column and row
       arr2[::-1,::-1]

* 
  Missing values and infinite

  .. code-block:: python

       np.nan # not a number
       np.inf # infinite

       # Replace nan and inf with -1. Don't use arr2 == np.nan
       missing_bool = np.isnan(arr2) | np.isinf(arr2)
       arr2[missing_bool] = -1  
       arr2

       arr2[(arr2 > 2) | (arr2 < 1)]

* 
  mean ,min ,max

  .. code-block:: python

       arr2.mean() , arr2.max() , arr2.min() # apply to whole array

       np.mean() , np.min() , np.max() , np.amin() , np.amax() ... # have axis args
       # axis = 0 column wise
       # axis = 1 row wise

       np.cumsum(arr2)

* 
  if a2 = a1 , then any changes in a2 will also be done on a1

  .. code-block:: python

       a2 = a1.copy()

* 
  reshaping

  .. code-block:: python

       arr2.reshape(4,3)
       arr2.reshape(-1,1) # reshape into 1 column only
       arr2.reshape(1,-1) # reshape into 1 row only

* 
  flatten and ravel


  * 
    using ravel will affect the parent

    .. code-block:: python

         b2 = arr2.ravel() # change b2 also change arr2

         b1 = arr2.flatten() # change b1 won't affect arr2

* 
  sequence

  .. code-block:: python

       # Lower limit is 0 be default
       print(np.arange(5))  

       # 0 to 9
       print(np.arange(0, 10))  

       # 0 to 9 with step of 2
       print(np.arange(0, 10, 2))  

       # 10 to 1, decreasing order
       print(np.arange(10, 0, -1))

       #> [0 1 2 3 4]
       #> [0 1 2 3 4 5 6 7 8 9]
       #> [0 2 4 6 8]
       #> [10  9  8  7  6  5  4  3  2  1]

       # Start at 1 and end at 50
       np.linspace(start=1, stop=50, num=10, dtype=int)
       # specify dtype = int because it will not equally spaced due to rounding

       #> array([ 1,  6, 11, 17, 22, 28, 33, 39, 44, 50]

       # Limit the number of digits after the decimal to 2
       np.set_printoptions(precision=2)  

       # Start at 10^1 and end at 10^50
       np.logspace(start=1, stop=50, num=10, base=10) 

       #> array([  1.00e+01,   2.78e+06,   7.74e+11,   2.15e+17,   5.99e+22,
       #>          1.67e+28,   4.64e+33,   1.29e+39,   3.59e+44,   1.00e+50])

       np.zeros([2,2])
       #> array([[ 0.,  0.],
       #>        [ 0.,  0.]])

       np.ones([2,2])
       #> array([[ 1.,  1.],
       #>        [ 1.,  1.]])

* 
  Repeating


  * 
    np.tile repeat a whole list or array n times while np.repeat repeats each item n times

    .. code-block:: python

       a = [1,2,3]

       np.tile(a,2) 
       # [1,2,3,1,2,3]

       np.repeat(a,2)
       # [1,1,2,2,3,3]

* 
  Random

  .. code-block:: python

       # Random numbers between [0,1) of shape 2,2
       print(np.random.rand(2,2))

       # Normal distribution with mean=0 and variance=1 of shape 2,2
       print(np.random.randn(2,2))

       # Random integers between [0, 10) of shape 2,2
       print(np.random.randint(0, 10, size=[2,2]))

       # One random number between [0,1)
       print(np.random.random())

       # Random numbers between [0,1) of shape 2,2
       print(np.random.random(size=[2,2]))

       # Pick 10 items from a given list, with equal probability
       print(np.random.choice(['a', 'e', 'i', 'o', 'u'], size=10))  

       # Pick 10 items from a given list with a predefined probability 'p'
       print(np.random.choice(['a', 'e', 'i', 'o', 'u'], size=10, p=[0.3, .1, 0.1, 0.4, 0.1]))  # picks more o's

       #> [[ 0.84  0.7 ]
       #>  [ 0.52  0.8 ]]

       #> [[-0.06 -1.55]
       #>  [ 0.47 -0.04]]

       #> [[4 0]
       #>  [8 7]]

       #> 0.08737272424956832

       #> [[ 0.45  0.78]
       #>  [ 0.03  0.74]]

       #> ['i' 'a' 'e' 'e' 'a' 'u' 'o' 'e' 'i' 'u']
       #> ['o' 'a' 'e' 'a' 'a' 'o' 'o' 'o' 'a' 'o']

* 
  random same value


  * 
    np.random.RandomState is created, all functions of np.random module becomes available to the created randomstate object

    .. code-block:: python

         # Create the random state
         rn = np.random.RandomState(100)

         # Create random numbers between [0,1) of shape 2,2
         print(rn.rand(2,2))

         #> [[ 0.54  0.28]
         #>  [ 0.42  0.84]]

         # Set the random seed
         np.random.seed(100)

         # Create random numbers between [0,1) of shape 2,2
         print(np.random.rand(2,2))

         #> [[ 0.54  0.28]
         #>  [ 0.42  0.84]]

* 
  get unique items and counts

  .. code-block:: python

       a = np.random.randint(1,10, 10)
       a
       array([8, 7, 1, 7, 7, 1, 8, 8, 4, 5])

       np.unique(a)
       array([1, 4, 5, 7, 8])

       np.unique(a, return_counts = True)
       (array([1, 4, 5, 7, 8]), array([2, 1, 1, 3, 3]))

       >>> unique, count = np.unique(a, return_counts = True)
       >>> unique
       array([1, 4, 5, 7, 8])
       >>> count
       array([2, 1, 1, 3, 3])

* 
  Get index location and take elements

  .. code-block:: python

       In [2]: a1 = np.array(np.random.randint(1,10,10))                               

       In [3]: a1                                                                      
       Out[3]: array([3, 1, 1, 1, 7, 8, 5, 3, 3, 2])

       In [4]: np.where(a1 > 3)                                                        
       Out[4]: (array([4, 5, 6]),)

       In [5]: a1.take(np.where(a1 > 3))                                               
       Out[5]: array([[7, 8, 5]])

       In [7]: np.where(a1 > 3, 'more than' , 'less than')                             
       Out[7]: 
       array(['less than', 'less than', 'less than', 'less than', 'more than',
              'more than', 'more than', 'less than', 'less than', 'less than'],
             dtype='<U9')

* 
  Get location of maximum and min values

  .. code-block:: python

       In [8]: a1                                                                      
       Out[8]: array([3, 1, 1, 1, 7, 8, 5, 3, 3, 2])

       In [9]: np.argmax(a1)                                                           
       Out[9]: 5

       In [10]: np.argmin(a1)                                                          
       Out[10]: 1

* 
  Import csv file from url and export csv

  .. code-block:: python

       data = np.genfromtxt(path, delimiter = ',', skip_header = 1, filling_values = __, dtype = 'float')
       # filing_values : replace if exists missng values or NaN

       data[:3] # see first 3 rows

       # Text column

       # data2 = np.genfromtxt(path, delimiter=',', skip_header=1, dtype='object')
       data2 = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=None)
       data2[:3]  # see first 3 rows

       # save array as a csv file
       np.savetxt("out.csv", data, delimiter=",")

* 
  numpy object : .npy, .npz

  .. code-block:: python

       # Save single numpy array object as .npy file
       np.save('myarray.npy', arr2d)  

       # Save multile numy arrays as a .npz file
       np.savez('array.npz', arr2d_f, arr2d_b)

       # Load a .npy file
       a = np.load('myarray.npy')
       b = np.load('myarray.npz')

* 
  concatenate two numpy arrays


  * np.concatenate
  * np.vstack / np.hstack
  * 
    np.r\ * / np.c* # use [ ]

    .. code-block:: python

         # Vertical Stack Equivalents (Row wise)
         np.concatenate([a, b], axis=0)  
         np.vstack([a,b])  
         np.r_[a,b]

         # Horizontal Stack Equivalents (Coliumn wise)
         np.concatenate([a, b], axis=1) 
         np.hstack([a,b])  
         np.c_[a,b]

* 
  sort array based on one or more columns

  .. code-block:: python

       In [23]: a                                                                      
       Out[23]: 
       array([[4, 2, 6],
              [3, 8, 6],
              [8, 1, 1],
              [8, 7, 4],
              [9, 6, 6],
              [5, 3, 2],
              [3, 7, 7]])

       In [24]: np.sort(a, axis = 0) # sort for every column                                              
       Out[24]: 
       array([[3, 1, 1],
              [3, 2, 2],
              [4, 3, 4],
              [5, 6, 6],
              [8, 7, 6],
              [8, 7, 6],
              [9, 8, 7]])

       In [25]: np.sort(a, axis = 1) # sort for every row                                             
       Out[25]: 
       array([[2, 4, 6],
              [3, 6, 8],
              [1, 1, 8],
              [4, 7, 8],
              [6, 6, 9],
              [2, 3, 5],
              [3, 7, 7]])

* 
  using argsort(x)

  .. code-block:: python

       In [3]: b                                                                       
       Out[3]: array([4, 2, 5, 9, 1, 2])

       In [4]: b.argsort()                                                             
       Out[4]: array([4, 1, 5, 0, 2, 3])

       In [5]: b[b.argsort()]                                                          
       Out[5]: array([1, 2, 2, 4, 5, 9])

       In [7]: a                                                                       
       Out[7]: 
       array([[5, 9, 3],
              [9, 1, 9],
              [6, 1, 9],
              [4, 8, 3],
              [4, 8, 8],
              [9, 1, 4],
              [4, 7, 7]])

       In [8]: a[a[:,0].argsort()]                                                     
       Out[8]: 
       array([[4, 8, 3],
              [4, 8, 8],
              [4, 7, 7],
              [5, 9, 3],
              [6, 1, 9],
              [9, 1, 9],
              [9, 1, 4]])

       In [11]: a[a[:,0].argsort()[::-1]]                                              
       Out[11]: 
       array([[9, 1, 4],
              [9, 1, 9],
              [6, 1, 9],
              [5, 9, 3],
              [4, 7, 7],
              [4, 8, 8],
              [4, 8, 3]])

* 
  Date

  .. code-block:: python

       # Create a datetime64 object
       In [23]: date64 = np.datetime64('2018-02-04 23:10:10')                          

       In [24]: date64                                                                 
       Out[24]: numpy.datetime64('2018-02-04T23:10:10')

       # Drop the time part from the datetime64 object
       dt64 = np.datetime64(date64, 'D')
       dt64


  * add number increase number of days
  * 
    timedelta

    .. code-block:: python

         In [30]: dt64                                                                   
         Out[30]: numpy.datetime64('2018-02-04')

         In [31]: dt64 + 10                                                              
         Out[31]: numpy.datetime64('2018-02-14')

         In [33]: dt64 + np.timedelta64(10, 'm')                                         
         Out[33]: numpy.datetime64('2018-02-04T00:10')

         In [34]: dt64 + np.timedelta64(10, 's')                                         
         Out[34]: numpy.datetime64('2018-02-04T00:00:10')

         In [35]: dt64 + np.timedelta64(10, 'ns') # 10 nanoseconds                                       
         Out[35]: numpy.datetime64('2018-02-04T00:00:00.000000010')

         # Convert np.datetime64 back to a string
         np.datetime_as_string(dt64)

  * 
    business days

    .. code-block:: python

         np.is_busday(dt64)) # check is business days

         print("Add 2 business days, rolling forward to nearest biz day: ", np.busday_offset(dt64, 2, roll='forward'))  
         print("Add 2 business days, rolling backward to nearest biz day: ", np.busday_offset(dt64, 2, roll='backward'))

  * 
    sequence of dates

    .. code-block:: python

         In [39]: np.arange(np.datetime64('2020-04-20'),np.datetime64('2020-04-30'))     
         Out[39]: 
         array(['2020-04-20', '2020-04-21', '2020-04-22', '2020-04-23',
                '2020-04-24', '2020-04-25', '2020-04-26', '2020-04-27',
                '2020-04-28', '2020-04-29'], dtype='datetime64[D]')

         In [40]: np.is_busday(np.arange(np.datetime64('2020-04-20'),np.datetime64('2020-
             ...: 04-30')))                                                              
         Out[40]: 
         array([ True,  True,  True,  True,  True, False, False,  True,  True,
                 True])

    .. code-block:: python

         # Convert np.datetime64 to datetime.datetime
         import datetime
         dt = dt64.tolist()

         print('Year: ', dt.year)  
         print('Day of month: ', dt.day)
         print('Month of year: ', dt.month)  
         print('Day of Week: ', dt.weekday())  # Sunday

* 
  normal function can't works on arrays

  .. code-block:: python

       # Define a scalar function
       def foo(x):
           if x % 2 == 1:
               return x**2
           else:
               return x/2

       # Vectorize foo(). Make it work on vectors.
       foo_v = np.vectorize(foo, otypes=[float])

* 
  Find differences of the max and min of every row


  * 
    numpy.apply_along_axis


    * 
      args :


      * function that works on a 1D vector fund1d
      * axis along which to apply func1d , 1 is row wise and 0 is column wise
      * array

      .. code-block:: python

         # Define func1d
         def max_minus_min(x):
           return np.max(x) - np.min(x)

         # Apply along the rows
         print('Row wise: ', np.apply_along_axis(max_minus_min, 1, arr=arr_x))

         # Apply along the columns
         print('Column wise: ', np.apply_along_axis(max_minus_min, 0, arr=arr_x))

* 
  searchsorted, find the way location to insert so the array will remain sorted


  * 
    gives the index position at which a number should be inserted in order to keep the array sorted

    .. code-block:: python

         In [46]: a                                                                      
         Out[46]: array([5, 6, 7, 8, 9])

         In [47]: np.searchsorted(a,4)                                                   
         Out[47]: 0

         In [48]: np.searchsorted(a,8)                                                   
         Out[48]: 3

* 
  create new axis to a existing array

  .. code-block:: python

       In [59]: x                                                                      
       Out[59]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

       In [60]: x.shape                                                                
       Out[60]: (10,)

       In [62]: x.ndim                                                                 
       Out[62]: 1

       In [63]: x = x[:,np.newaxis]                                                    

       In [64]: x                                                                      
       Out[64]: 
       array([[0],
              [1],
              [2],
              [3],
              [4],
              [5],
              [6],
              [7],
              [8],
              [9]])

       In [65]: x.ndim                                                                 
       Out[65]: 2

       In [66]: x.shape                                                                
       Out[66]: (10, 1)

       In [72]: x = x[np.newaxis,:]                                                    

       In [73]: x                                                                      
       Out[73]: array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

       In [74]: x.shape                                                                
       Out[74]: (1, 10)

       In [75]: x.ndim                                                                 
       Out[75]: 2

* 

.. code-block:: python

   np.floor() : # get the floor of the value if 2.3 -> 2


* a
