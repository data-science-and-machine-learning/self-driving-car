import scipy.misc
import random
import imageio
from PIL import Image
import numpy as np

xs = []
ys = []

#drive/My Drive/Autopilot-TensorFlow/driving_dataset/driving_dataset/

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0
counter= 0
#read data.txt
with open("drive/My Drive/Autopilot-TensorFlow/Files/data.txt") as f:
    for line in f:
        full_path= "driving_dataset/driving_dataset/" + line.split()[0]
        
        radian= float(line.split()[1]) * scipy.pi / 180 
        ys.append(radian)
        xs.append(full_path)
        
        
#get number of images
num_images = len(xs)
print(num_images)

#Assignment job : Changing split into 70 and 30
train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(np.array(Image.fromarray(imageio.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:]).resize([200, 66]), dtype=float) / 255.0) #.imresize()
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
        #print("xshape", np.shape(x_out))
        #print("imageShape", np.shape(np.array(Image.fromarray(imageio.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:]).resize([66, 200]), dtype=float) / 255.0))
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(np.array(Image.fromarray(imageio.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:]).resize([200, 66]), dtype=float) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out