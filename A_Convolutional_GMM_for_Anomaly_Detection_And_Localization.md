
In this work, we shall define anomalies to be events that are unlikely to occur based on what we have learned from the training dataset. One can hence think of it as a one class learning problem and try learning the underlying distribution for the data. Gaussian Mixture Models(GMM) are proven to approximate any kind of probability distribution by choosing the  appropriate number of mixture components. In practice some sort of dimensionality reduction technique like PCA, SVD, 
autoencoders etc is carried out before passing it to a GMM.  Such techniques need not necessarily learn the representation that best maximizes the likelihood of observed samples while minimizing the rest.  In recent years convolutional neural networks are shown to have better representational capabilities compared to other popular techniques. 

In this work, the GMM is used as the output layer of a 3D CNN that takes spatio-temporal volume(SVOI) as input and the parameters are learned via backpropagation. The loss function is chosen in a manner that maximizes the likelihood of observed samples and minimizes the likelihood of potential unobserved samples. This method assumes that the patterns that are generally observed in a SVOI are very similar and any noise with a non negligible variance if added to the SVOI would 
make the pattern imitate an outlier. This is justified as one of the maps contain optical flow information and when noise is added to it, it will most likely represent a physically impossible motion behaviour.
Different transformations of the input can lead to qualitatively different representations. In this work the input is a 2 Channel image where the 1st channel captures spatial saliency and the 2nd channel captures optical flow. This transformation is chosen under the following assumption that anomalies are generally attention capturing and their motion patterns might be different from what is commonly seen.

The novel aspect of this work is the following:
    - The joint learning of GMM and 3D CNN.
    - The minimax training procedure where we maximize the likelihood of least probable observation and minimize the      likelihood of the most probable noisy observation. These noisy observations act as counter-examples during the training phase.


```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 19:17:41 2017

@author: QuantumMole
"""
import tensorflow as tf
import numpy as np
from random import shuffle
import cv2
from sklearn.metrics import roc_auc_score as auc_score
from skimage import io,filters,transform

from matplotlib import pyplot as plt
keep_prob = tf.placeholder(tf.float32)
mode = tf.placeholder(tf.bool)

tf.logging.set_verbosity(tf.logging.INFO)

STEPS = 1001
DATA_DIR='.'
stride = (1,1)
batch_size = 256

VOLUME_HEIGHT,VOLUME_WIDTH,VOLUME_DEPTH = 32,32,7
keep_prob = tf.placeholder(tf.float32)
mode = tf.placeholder(tf.bool)

tf.logging.set_verbosity(tf.logging.WARN)
np.random.seed(42)

gen = lambda x,y : [i for i in range(x,y)]

def weight_variable(shape):    
    initial = tf.truncated_normal(shape, stddev=0.1)    
    return tf.Variable(initial)

def bias_variable(shape) :
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def max_pool_2x2_3D(x,d,size):
    d1,m,n = size
    return tf.layers.max_pooling3d(inputs=x, pool_size=[d,m, n], strides=(d1,m,n))

def max_pool(x,stride):
    return tf.layers.max_pooling2d(inputs=x, pool_size=stride, strides=stride)

def conv3_layer(input,num_filters,shape,stride,scale_l1 = 0.0,scale_l2=1.0):
    regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1,scale_l2)
    return tf.layers.conv3d(inputs = input,kernel_size=shape,
                     filters = num_filters,activation=tf.nn.relu,
                     padding='same',use_bias=True,strides=stride,kernel_regularizer = regularizer)

def conv2_layer(input,num_filters,shape,stride):
    y= tf.layers.conv2d(inputs = input,kernel_size=shape,
                     filters = num_filters,activation=None,
                     padding='same',use_bias=True,strides=stride)
    return tf.maximum(-y,y)

def conv2t_layer(input,num_filters,shape,stride):
    y = tf.layers.conv2d_transpose(inputs = input,kernel_size=shape,
                     filters = num_filters,activation=None,
                     padding='same',use_bias=True,strides=stride)
    return tf.maximum(-y,y)

def full_layer(input, size,activation):
 return tf.layers.dense(inputs=input, units=size, activation=activation)

def droput_layer(input,keep_prob,mode) :
    return tf.layers.dropout(inputs=input, rate=1-keep_prob, 
                    training = mode)
def batchnorm_layer(input) :
    return tf.layers.batch_normalization(inputs=input)
```


```python
SHUFFLE_FLAG = True
curr_index = 0
input_final = []
def getBatch(input,size) :
    global SHUFFLE_FLAG
    global curr_index
    global input_final,batch_size
    min_size =  min([len(input),size])
    if SHUFFLE_FLAG :
        shuffle(input)
        input_final = input
        SHUFFLE_FLAG = False
    curr_index += np.random.randint(1,batch_size)
    curr_index = curr_index %len(input_final)
    return input_final[curr_index:curr_index+min_size]


def linearize_frame(x) :
        y = x.reshape((1,x.shape[0]*x.shape[1]))
        return y

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def reconstructImagesPatches(patches,has_anomaly,o_h,o_w) :
    [length,frames,h,w] = patches.shape
    print(length,h,w)
    batches = length*h*w//o_h//o_w
    print(batches)
    video = np.zeros((batches,o_h,o_w))
    print(video.shape)
    num_regions = o_h*o_w//h//w
    num_horizontal_regions = o_w//w
    for index in range(length) :
        batch_num = index // num_regions
        i = (index - batch_num*num_regions)//num_horizontal_regions
        j = (index - batch_num*num_regions - i*num_horizontal_regions)%10
        if has_anomaly[index] == 1 :
            z =  patches[index,frames-1,:,:]
            z[0:4,:] = 0
            z[h-4:h,:] = 0
            z[:,0:4] = 0
            z[:,w-4:w] =w
            video[batch_num,i*h:(i+1)*h,j*w:(j+1)*w] = z
        else :    
            video[batch_num,i*h:(i+1)*h,j*w:(j+1)*w] = patches[index,frames-1,:,:]
    print(video.shape)
    return video

def createSpatioTemporalPatches(imageArray,x_width,y_width,num_frames,index) :
    [batches,h,w] = imageArray.shape[0],imageArray.shape[1],imageArray.shape[2]
    spatio_patches = []
    for frame in range(0,batches-num_frames) :
        for i in range(0,h//y_width) :
            for j in range(0,w//x_width) :
                spatio_patches.append((index,[gen(frame,frame+num_frames),gen(i*y_width,(i+1)*y_width),gen(j*x_width,(j+1)*x_width)]))
    return spatio_patches


def getData(train=True) :
    import os
    all_folders = [x for x in filter(lambda x : "Train" in x ,sorted(os.listdir('./UCSDped2/Train/')))]
    input_path = r"/home/vsl4/Mani/datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/{}/{}"
    file_path = input_path +"/{}"
    r_v = []
    images =[]
    for index,folder in enumerate(all_folders) :
        flow_files = [transform.resize(io.imread(file_path.format(folder,"flow",file)),(240,360)) for file in sorted(os.listdir(input_path.format(folder,"flow")))]
        salient_files = [transform.resize(io.imread(file_path.format(folder,"saliency",file)),(240,360)) for file in sorted(os.listdir(input_path.format(folder,"saliency")))][1:]
        maps = np.array([np.dstack([salient_files[i],flow_files[i]]) for i in range(0,len(flow_files))])
        r_v = r_v +  createSpatioTemporalPatches(maps,VOLUME_HEIGHT,VOLUME_WIDTH,VOLUME_DEPTH,index)
        images.append(maps)
    return np.array(images)+0.0001,r_v

def isAnomaly(patch_gt,threshold) :
    [f,h,w] = patch_gt.shape
    return np.sum(patch_gt) > threshold*h*w

def createTestSpatioTemporalPatches(imageArray,x_width,y_width,num_frames) :
    [batches,h,w] = imageArray.shape[0],imageArray.shape[1],imageArray.shape[2]
    spatio_patches = []
    for frame in range(0,batches-num_frames) :
        for i in range(0,h//y_width) :
            for j in range(0,w//x_width) :
                spatio_patches.append(imageArray[frame:frame+num_frames,i*y_width:(i+1)*y_width,j*x_width:(j+1)*x_width])
    return np.array(spatio_patches)

def getTestData(i) :
    import os
    all_folders = os.listdir('./UCSDped2/Test/')
    input_path = r"/home/vsl4/Mani/datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/{}/{}"
    file_path = input_path +"/{}"
    dirs = sorted([x for x in all_folders if x[-1]=='t'])

    positives_image = []
    negatives_image = []

    for folder in dirs[i:i+1] :
        folderr = folder.replace("_gt","")
        ground_truths = ["./UCSDped2/Test/{}/{}".format(folder,x) for x in os.listdir('./UCSDped2/Test/{}'.format(folder)) if x[-1] == 'p']
        ground_truths = sorted(ground_truths)
        flow_files = [transform.resize(io.imread(file_path.format(folderr,"flow",file)),(240,360)) for file in sorted(os.listdir(input_path.format(folderr,"flow")))]
        salient_files = [transform.resize(io.imread(file_path.format(folderr,"saliency",file)),(240,360)) for file in sorted(os.listdir(input_path.format(folderr,"saliency")))][1:]
        maps = np.array([np.dstack([salient_files[i],flow_files[i]]) for i in range(0,len(flow_files))])
        images = maps[:,:,:,:]
        patches_image = createTestSpatioTemporalPatches(images,32,32,7)
        video_gt = np.array([transform.resize(io.imread(image),(240,360))  for image in ground_truths])
        images_gt = video_gt[:,:,:]
        patches_image_gt = createTestSpatioTemporalPatches(images_gt[4:-3],32,32,1)
        has_anomaly_image = np.array([isAnomaly(patch,0.4) for patch in patches_image_gt])

        positive_entries_image = has_anomaly_image
        negative_entries_image = ~has_anomaly_image
        positives_image.append(patches_image[positive_entries_image])
        negatives_image.append(patches_image[negative_entries_image])     

    positives_image = np.vstack(positives_image)
    negatives_image = np.vstack(negatives_image)
    
    return positives_image,negatives_image
```


```python
  
def network(image_batch,num_components,FINAL_FILTERS) :
    conv_1 = batchnorm_layer(conv3_layer(image_batch,FINAL_FILTERS/32,(3,3,3),(1,1,1)))
    conv_11 = batchnorm_layer(conv3_layer(conv_1,FINAL_FILTERS/16,(3,3,3),(1,1,1)))
    pool_1 = max_pool_2x2_3D(conv_11,3,(2,2,2))  #16x16
    pool_11 = max_pool_2x2_3D(pool_1,3,(3,2,2))
    print(pool_1)
    conv_2 = batchnorm_layer(conv3_layer(pool_1,FINAL_FILTERS/32,(3,3,3),(1,1,1)))
    conv_22 = batchnorm_layer(conv3_layer(conv_2,FINAL_FILTERS/16,(3,3,3),(1,1,1)))
    pool_2 = max_pool_2x2_3D(conv_22,3,(3,2,2))
    print(pool_2) #8x8
    conv_3 = batchnorm_layer(conv3_layer(pool_2,FINAL_FILTERS/16,(1,3,3),(1,1,1)))
    conv_33 = batchnorm_layer(conv3_layer(conv_3,FINAL_FILTERS/8,(1,5,5),(1,1,1)))
    conv_331 = tf.concat([pool_11,pool_2,conv_33],4)
    pool_3 = max_pool_2x2_3D(conv_33,1,(1,2,2))
    pool_33 = max_pool_2x2_3D(conv_331,1,(1,2,2))
    print(pool_3) #4x4
    conv_4 = batchnorm_layer(conv3_layer(pool_3,FINAL_FILTERS/8,(1,2,2),(1,1,1)))
    conv_44 = batchnorm_layer(conv3_layer(conv_4,FINAL_FILTERS/4,(1,2,2),(1,1,1)))
    conv_441 = tf.concat([pool_33,conv_44],4)
    pool_4 = batchnorm_layer(conv3_layer(conv_441,FINAL_FILTERS,(1,4,4),(1,4,4),1.0,0.0))
    print(pool_4)
    fc1_flat = tf.reshape(pool_4,(-1,FINAL_FILTERS))
    print(fc1_flat)
    weights_unnormalized = tf.exp(weight_variable((1,num_components)))
    normalizer = tf.reduce_sum(weights_unnormalized)
    mixture_weights = weights_unnormalized/normalizer    
    var_n = weight_variable((1,FINAL_FILTERS))
    var = tf.square(var_n) + tf.constant(value=0.000001)
    sd = tf.sqrt(var)
    gaussians = [] 
    for i in range(num_components) : 
         mean =  weight_variable((1,FINAL_FILTERS))
         gaussian = tf.reduce_max(tf.abs(tf.multiply(fc1_flat - mean,1.0/sd)),axis = 1)
         gaussians.append(gaussian)
    concatted_gaussians = tf.reshape(tf.stack(gaussians,axis=1),(-1,num_components))
    print(concatted_gaussians)
    y_conv = tf.multiply(mixture_weights,concatted_gaussians)
    print(y_conv)
    return concatted_gaussians,y_conv
```


```python
images,image_indices = getData()
m,n,a,b = images[0].shape
shuffle(image_indices)
trainr_set = image_indices[0:int(0.75*len(image_indices))]
validationr_set = image_indices[int(0.75*len(image_indices)):]


eta = tf.placeholder(tf.float32,name = "eta")
image_batch = tf.placeholder(tf.float32,shape=(None,VOLUME_DEPTH,VOLUME_HEIGHT,VOLUME_WIDTH,b),name = "image_input")
gauss,image_likelihoods = network(image_batch,10,128)
expected_distance = tf.log(tf.reduce_sum(image_likelihoods,axis = 1))
image_entropy = tf.reduce_max(expected_distance)
image_min_likelihood = tf.placeholder(tf.float32,name = "image_min_likelihood")
image_max_likelihood = tf.placeholder(tf.float32,name = "image_max_likelihood")

image_loss = eta*tf.reduce_min(expected_distance)


train1_step = tf.train.AdamOptimizer(1e-4).minimize(image_entropy)

train_step = tf.train.AdamOptimizer(1e-4).minimize(-1*image_loss)

def getimageBatch(indexes) :
    batch = []
    for (index,z) in indexes :

        batch.append(images[index][z[0]][:,z[1]][:,:,z[2]])
    return np.array(batch)
#HMM Training Part

train_noisy_max =  tf.summary.scalar('image_noisy_max', image_loss/eta)
train_clean_min =  tf.summary.scalar('image_clean_min', image_entropy)
diff =     tf.summary.scalar('diff_min_max', image_max_likelihood -  image_min_likelihood)
validation_loss = tf.summary.scalar('validation_image_min', image_entropy)
saver = tf.train.Saver(max_to_keep=0)
LOG_DIR="/home/vsl4/Mani/tensorflow/CNN_HMMGMM/ped2_joint"
```

    /home/vsl4/anaconda3/lib/python3.6/site-packages/skimage/transform/_warps.py:84: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "


    Tensor("max_pooling3d/MaxPool3D:0", shape=(?, 3, 16, 16, 8), dtype=float32)
    Tensor("max_pooling3d_3/MaxPool3D:0", shape=(?, 1, 8, 8, 8), dtype=float32)
    Tensor("max_pooling3d_4/MaxPool3D:0", shape=(?, 1, 4, 4, 16), dtype=float32)
    Tensor("batch_normalization_9/batchnorm/add_1:0", shape=(?, 1, 1, 1, 128), dtype=float32)
    Tensor("Reshape:0", shape=(?, 128), dtype=float32)
    Tensor("Reshape_1:0", shape=(?, 10), dtype=float32)
    Tensor("Mul_10:0", shape=(?, 10), dtype=float32)



```python
print("Beginning training")
deviation = 0.2
for dev in range(0,5) :
    damping = 0.01
    for damp in range(0,5) :
        curr_auc_score = 0.0
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('{}/graphs'.format(LOG_DIR),sess.graph)
            sess.run(tf.global_variables_initializer())
            for i in range(STEPS):
                train_data_indexes = getBatch(trainr_set,batch_size)
                train_data_batch = getimageBatch(train_data_indexes)
                train_data_batch_noisy = train_data_batch + np.random.normal(scale = deviation,size = train_data_batch.shape )
                _,rmin,tcm = sess.run([train1_step,image_entropy,train_clean_min], feed_dict={image_batch : train_data_batch})
                _,rmax,tnm = sess.run([train_step,image_loss,train_noisy_max], feed_dict={image_batch : train_data_batch_noisy,eta: damping,image_min_likelihood : rmin})          
                dmm = sess.run(diff,feed_dict={image_max_likelihood:rmax/damping,image_min_likelihood:rmin})

                if i%10 == 0 :
                    writer.add_summary(tcm,i)
                    writer.add_summary(tnm,i)
                    writer.add_summary(dmm,i)
                    p_rt = []
                    n_rt = []
                    for j in range(0,1) :
                        p_r,n_r = getTestData(j)
                        m,n,a,c,b = p_r.shape
                        for i in range(0,p_r.shape[0],250):
                            vals = sess.run(expected_distance, feed_dict={image_batch : p_r[i:i+250],eta: 0.5})
                            p_rt = p_rt + list(vals.ravel())

                        for i in range(0,n_r.shape[0],250):
                            vals = sess.run(expected_distance, feed_dict={image_batch : n_r[i:i+250],eta: 0.5})
                            n_rt = n_rt + list(vals.ravel())
                    auc_score_now = auc_score([1]*len(p_rt)+[0]*len(n_rt),p_rt+n_rt)
                    if auc_score_now > curr_auc_score :
                        saver.save(sess, "{}/model_ped2_salient_flow.ckpt".format(LOG_DIR))
                        curr_auc_score = auc_score_now
        p_rt = []
        n_rt = []
        with tf.Session() as sess:
            saver.restore(sess,"{}/model_ped2_salient_flow.ckpt".format(LOG_DIR))
            for j in range(1,11) :
                p_r,n_r = getTestData(j)
                m,n,a,c,b = p_r.shape

                for i in range(0,p_r.shape[0],250):
                    vals = sess.run(expected_distance, feed_dict={image_batch : p_r[i:i+250],eta: damping})
                    p_rt = p_rt + list(vals.ravel())

                for i in range(0,n_r.shape[0],250):
                    vals = sess.run(expected_distance, feed_dict={image_batch : n_r[i:i+250],eta: damping})
                    n_rt = n_rt + list(vals.ravel())

        print("auc score on test data for {} std and damping rate {} is {} and its validation auc was {}".format(deviation,damping,auc_score([1]*len(p_rt)+[0]*len(n_rt),p_rt+n_rt),curr_auc_score))
        damping = damping*10
    deviation = deviation+0.2
```

    Beginning training


    /home/vsl4/anaconda3/lib/python3.6/site-packages/skimage/transform/_warps.py:84: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "


    auc score on test data for 0.2 std and damping rate 0.01 is 0.7160659240882586 and its validation auc was 0.8157125464659905
    auc score on test data for 0.2 std and damping rate 0.1 is 0.6083442512846282 and its validation auc was 0.5047105303145418
    auc score on test data for 0.2 std and damping rate 1.0 is 0.847008993415942 and its validation auc was 0.9083313567142661
    auc score on test data for 0.2 std and damping rate 10.0 is 0.9283476112369272 and its validation auc was 0.8891941324062991
    auc score on test data for 0.2 std and damping rate 100.0 is 0.8523730336273729 and its validation auc was 0.7444968145548345
    auc score on test data for 0.4 std and damping rate 0.01 is 0.9159491683140094 and its validation auc was 0.9045367113753191
    auc score on test data for 0.4 std and damping rate 0.1 is 0.8217168204477021 and its validation auc was 0.7999849900489583
    auc score on test data for 0.4 std and damping rate 1.0 is 0.7019415535358556 and its validation auc was 0.7411399594546014
    auc score on test data for 0.4 std and damping rate 10.0 is 0.862907813808193 and its validation auc was 0.7929905381716027
    auc score on test data for 0.4 std and damping rate 100.0 is 0.8654633428280702 and its validation auc was 0.78683645824454
    auc score on test data for 0.6000000000000001 std and damping rate 0.01 is 0.9462696768004181 and its validation auc was 0.9303839397230015
    auc score on test data for 0.6000000000000001 std and damping rate 0.1 is 0.8085927207999486 and its validation auc was 0.848827556046416
    auc score on test data for 0.6000000000000001 std and damping rate 1.0 is 0.8594209436978206 and its validation auc was 0.7887641251051623
    auc score on test data for 0.6000000000000001 std and damping rate 10.0 is 0.8085800176048593 and its validation auc was 0.7845117689134649
    auc score on test data for 0.6000000000000001 std and damping rate 100.0 is 0.7667487878812027 and its validation auc was 0.7718019538879479
    auc score on test data for 0.8 std and damping rate 0.01 is 0.9494547883092047 and its validation auc was 0.897299042698678
    auc score on test data for 0.8 std and damping rate 0.1 is 0.9535029129201968 and its validation auc was 0.9733142527824002
    auc score on test data for 0.8 std and damping rate 1.0 is 0.9068239810424022 and its validation auc was 0.87384692073634
    auc score on test data for 0.8 std and damping rate 10.0 is 0.6698782637016656 and its validation auc was 0.7055668387560642
    auc score on test data for 0.8 std and damping rate 100.0 is 0.8819718543267627 and its validation auc was 0.8398665226205522
    auc score on test data for 1.0 std and damping rate 0.01 is 0.8255109278502931 and its validation auc was 0.825940160328514
    auc score on test data for 1.0 std and damping rate 0.1 is 0.8963695679612481 and its validation auc was 0.8522603874420449
    auc score on test data for 1.0 std and damping rate 1.0 is 0.8568635002491144 and its validation auc was 0.7474765677986517
    auc score on test data for 1.0 std and damping rate 10.0 is 0.8857284747225665 and its validation auc was 0.7916498345199225
    auc score on test data for 1.0 std and damping rate 100.0 is 0.9133744872236104 and its validation auc was 0.9258828074908921


The AUC for anomaly localization is 0.95 which is comparable to the state of the art for UCSD Ped2
