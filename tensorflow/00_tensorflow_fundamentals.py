## udemy tensorflow course
## jyk4100
## 2021-05-31

## https://prudential.udemy.com/course/tensorflow-developer-certificate-machine-learning-zero-to-mastery/
## section 2 tensor flow fundamentals

import tensorflow as tf
print(tf.__version__)

## scalar, vector, matrix, tensor
## scalar rank 0 tensor just a number with no dimension
scalar = tf.constant(7)
scalar
scalar.ndim

## vector dimension 1
vector = tf.constant([10, 10])
vector
vector.ndim

## matrix
mat = tf.constant([[10, 7],
                   [7, 10]])
mat
mat.ndim
## int32 or float32 but could define as float16 such as
mat2 = tf.constant([[10., 7.],
                    [3., 2.],
                    [8., 9.]], dtype=tf.float16) # specify the datatype with 'dtype'
mat2


## tensor
tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]],
                      [[13, 14, 15],
                       [16, 17, 18]]])
tensor
tensor.shape ## 3 2 by 3 matrix/block matrix
tensor.ndim 
## 3 dimensional tensor can be arbitary n dimension 3
## e.g. image represented in (224, 224, 3, 32)
## 224, 224 are the height and width of the images in pixels.
## 3 is the number of colour channels of the image (red, green blue).
## 32 is the batch size (the number of images a neural network sees at any one time).

## creating tensors with tf.Variable()
## tf.constant is immutable/fixed

changeable_tensor = tf.Variable([10, 7])
unchangeable_tensor = tf.constant([10, 7])
changeable_tensor
## 
changeable_tensor[0] = 7 ## can't change by index
changeable_tensor[0].assign(7)
unchangeable_tensor[0].assign(7) ## can't change/immutable ## no attribute assign
unchangleable_tensor

## random tensors tf.random.Generator class as in random initailization
## randomseed
rand_1 = tf.random.Generator.from_seed(42) # set the seed for reproducibility
rand_1 = rand_1.normal(shape=(3, 2)) # create tensor from a normal distribution 
rand_2 = tf.random.Generator.from_seed(42)
rand_2 = rand_2.normal(shape=(3, 2))

## random seed works and element wise comparison also works
rand_1 == rand_2
## random seed changes (psuedo) rng differ
random_3 = tf.random.Generator.from_seed(42)
random_3 = random_3.normal(shape=(3, 2))
random_4 = tf.random.Generator.from_seed(11)
random_4 = random_4.normal(shape=(3, 2))

## shuffling order of tensor
not_shuffled = tf.constant([[10, 7], [3, 4], [2, 5]])
tf.random.shuffle(not_shuffled) ## "rows-wise" shuffle
tf.random.shuffle(not_shuffled, seed=42) ## setting seed
## seed does'st work? tf.random.set_seed() documentation -> both global and operation seed are used
tf.random.set_seed(42) ## global seed
tf.random.shuffle(seed=42) ## operation seed

## to shuffle reproducible
tf.random.set_seed(42)
tf.random.shuffle(not_shuffled, seed=42)
# Set the global random seed
tf.random.set_seed(42) # if you comment this out you'll get different results
tf.random.shuffle(not_shuffled)

## other ways to create tensors
tf.ones(shape=(3, 2))
tf.zeros(shape=(3, 2, 1))

## numpy array to tensor tensor can run on GPU
import numpy as np
np_vec = np.arange(1, 25, dtype=np.int32)
vec = tf.constant(np_vec, shape=[2, 4, 3]) 
## shape total (2*4*3) has to match the number of elements in the array
vec

## shape, rank, size
## shape: number of elements for each dimension
## rank: number of dimensions
## dimension (axis): a particular dimension of a tensor
## size: total number of items in the tensor
## e.g. making sure the shape of your image tensors are the same shape as your models input layer

## 4 by 5 matrix 2 by 3?
rank_4_tensor = tf.zeros([2, 3, 4, 5])
rank_4_tensor

## attributes of tensor
print("Datatype of every element:", rank_4_tensor.dtype)
## string concat? how is this possible lol
type(("datatype", rank_4_tensor.dtype))
print("temp", 123, 123)
rank_4_tensor.ndim ## numpy
rank_4_tensor.shape
rank_4_tensor.shape[0] ## axis 0
rank_4_tensor.shape[-1] ## axis 4
tf.size(rank_4_tensor).numpy() ## size

## indexing tensors
## 3 "layers" of 2 by 5 matrix
r3 = tf.constant(range(0,30), shape=(3,2,5)) # tf.constant( np.array(range(0,10)), shape=(2,5))
r3[:2, :2, :3] ## first 2 layers of 2 by 3 matrix
r3[1, :2, :3] ## 2nd layer 2 by 3 matrix
r3[2, :, :] ## 3rd layer 2 by 5 matrix


## 2 by 2 block matrix of 2 by 3 matrix ? ## nope
## 2 2 by 3 matrix with depth 2?
r4 = tf.constant(range(0,24), shape=(2,2,2,3)) # tf.constant( np.array(range(0,10)), shape=(2,5))
r4 = tf.constant(range(0,36), shape=(3,2,2,3))
r4

## first 2 items of each dimension 
rank_4_tensor
rank_4_tensor[:2, :2, :2, :2]
r4[:2, :2, :2, :2]
r4[:, :3, :3, :3]

## get the dimension from each index except for the final one ??
rank_4_tensor.shape
rank_4_tensor[:1, :1, :1, :]