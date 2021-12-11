from sklearn.model_selection import StratifiedShuffleSplit
import cv2

def single_stratified_split(X, Y, test_size=.2, random_state=1234):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    split_ix = splitter.split(X,Y)
    for train_ix, test_ix in split_ix:
        X_train = X[train_ix]
        Y_train = Y[train_ix]
        X_test = X[test_ix]
        Y_test = Y[test_ix]
    return X_train, Y_train, X_test, Y_test

# class MiRed3(Model): # conv
#   def __init__(self):
#     super(MiRed3, self).__init__()
#     self.conv1=CapaConvolucional(kernel_dim=3,input_chanel=3,output_chanel=32,padding='VALID')
#     #self.maxpool=MaxPool(2,2)
#     self.redim=Redimensionar(30,30,32)
#     self.densa1=CapaDensa(input_dim=30*30*32,output_dim=128,activation="relu")
#     self.densa2=CapaDensa(input_dim=128,output_dim=10,activation="softmax")

#   def call(self,x):
#     #print("orig",x.shape)   
#     x = self.conv1(x)
#     #print("conv",x.shape)
#     x = self.redim(x)
#     #print("redim",x.shape)
  
#     x = self.densa1(x)
#     #print("densa",x.shape)
#     return self.densa2(x)

# class MiRed4(Model): # fc
#   def __init__(self):
#     super(MiRed4, self).__init__()
#     self.conv1=CapaConvolucional(kernel_dim=3,input_chanel=3,output_chanel=32,padding='VALID')
#     #self.maxpool=MaxPool(2,2)
#     self.redim=Redimensionar(30,30,32)
#     self.densa1=CapaDensa(input_dim=30*30*32,output_dim=128,activation="relu")
#     self.densa2=CapaDensa(input_dim=128,output_dim=64,activation="relu")
#     self.densa3=CapaDensa(input_dim=64,output_dim=10,activation="softmax")

#   def call(self,x):
#     #print("orig",x.shape)   
#     x = self.conv1(x)
#     #print("conv",x.shape)
#     x = self.redim(x)
#     #print("redim",x.shape)
  
#     x = self.densa1(x)
#     x = self.densa2(x)
#     #print("densa",x.shape)
#     return self.densa3(x)

#   class MiRed5(Model): # filtros
#     def __init__(self):
#       super(MiRed5, self).__init__()
#       self.conv1=CapaConvolucional(kernel_dim=3,input_chanel=3,output_chanel=32,padding='VALID')
#       #self.maxpool=MaxPool(2,2)
#       self.redim=Redimensionar(30,30,32)
#       self.densa1=CapaDensa(input_dim=30*30*32,output_dim=128,activation="relu")
#       self.densa2=CapaDensa(input_dim=128,output_dim=10,activation="softmax")

#     def call(self,x):
#       #print("orig",x.shape)   
#       x = self.conv1(x)
#       #print("conv",x.shape)
#       x = self.redim(x)
#       #print("redim",x.shape)
    
#       x = self.densa1(x)
#       #print("densa",x.shape)
#       return self.densa2(x)


# class CapaConvolucional(keras.layers.Layer):
#     def __init__(self, kernel_dim=3,input_chanel=3, output_chanel=16 ,stride=1 ,padding='VALID', batchnorm=False):
#         super(CapaConvolucional, self).__init__()

#         self.stride=stride
#         self.padding=padding
#         print(output_chanel)
#         w_init =tf.keras.initializers.GlorotUniform()
#         self.w = tf.Variable(
#             initial_value=w_init(shape=(kernel_dim, kernel_dim,input_chanel,output_chanel), dtype="float32"),
#             trainable=True,
#         )
#         #print("w",self.w.shape)
#         b_init =tf.keras.initializers.Zeros()
#         self.b = tf.Variable(
#             initial_value=b_init(shape=(output_chanel,), dtype="float32"), trainable=True
#         )
#         self.batchnorm = batchnorm
#         if self.batchnorm:
#           self.b_n = tf.keras.layers.BatchNormalization()

#         #print("b",self.b.shape)
      

#     def call(self, inputs):
#       #tf.nn.conv2d(   input, filters, strides, padding, data_format='NHWC', dilations=None,   name=None)
#       convolucion=tf.nn.conv2d(inputs, self.w, strides=[1,  self.stride, self.stride, 1], padding=self.padding)
#       #print("conv",convolucion.shape)
#       x = tf.nn.relu(convolucion + self.b)
#       if self.batchnorm:
#         x = self.b_n(x)
#       return x