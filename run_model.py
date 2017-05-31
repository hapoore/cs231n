import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import csv
import model
import conv_model_3d
import random
import os.path
import os
from scipy.misc import imread

def read_csv(filename, num_train, num_val):
    classes = {}
    class_to_number = {}
    number_to_class = {}
    unique_genres = set()
    filenames = []
    
    random.seed(12345)
    
    with open(filename, 'rU') as csvfile:
        vidreader = csv.DictReader(csvfile)
        for row in vidreader:
            unique_genres.add(row['Genre'])
            filenames.append(row['FileName'])
        
    counter = 0
    for genre in unique_genres:
        class_to_number[genre] = counter
        number_to_class[counter] = genre
        counter += 1
    print(class_to_number)
            
    with open(filename, 'rU') as csvfile:
        vidreader = csv.DictReader(csvfile)
        for row in vidreader:
            classes[row['FileName']] = class_to_number[row['Genre']]

    random.shuffle(filenames)
#    print(len(filenames))
    filenames_train = filenames[:num_train]
    filenames_val = filenames[num_train:num_train+num_val]
    filenames_test = filenames[num_train + num_val:]
    return classes, class_to_number, number_to_class, filenames_train, filenames_val, filenames_test
    
def get_batch_frames(filenames, batch_size, train_indices,
                     number_to_class, classes, batch_num, num_frames, crop_dim, mean_img=None):
    # generate indices for the batch
#    print('starting read files')
    start_idx = (batch_num*batch_size)%len(filenames)
    idx = train_indices[start_idx:start_idx+batch_size]
    #print(idx)
    #per-batch vars
    batch_frames = []
    batch_labels = []
    
    for j in idx:

        directory = ('../output_frames/' + number_to_class[classes[filenames[j]]]
                    + '/' + filenames[j].split('.')[0])
        total_frames = len([name for name in os.listdir(directory) if os.path.isfile(name)])
        interval = int(math.floor(total_frames/num_frames))
        frames = []
        for frame_number in range(num_frames):
            frame_path = directory + '/frame' + str(interval*frame_number+1) + '.jpg'
            frame = imread(frame_path)
            height = frame.shape[0]
            width = frame.shape[1]
            vert_indent = int(math.ceil((height - crop_dim)/2))
            horiz_indent = int(math.ceil((width - crop_dim)/2))
            frame = frame[vert_indent:vert_indent+crop_dim, 
                          horiz_indent:horiz_indent+crop_dim, :]  
            frame = frame.astype('float32') / 256   
            frames.append(frame)
        batch_labels.append(classes[filenames[j]])
        batch_frames.append(frames)
        
#        filepath = ('../SVW/Videos/' + number_to_class[classes[filenames[j]]]
#                    + '/' + filenames[j])
#        cap = cv2.VideoCapture(filepath)
#        ret, frame = cap.read()
#        #print(filepath)
#        frame_count = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
#        #print(frame_count)
#        interval = int(math.floor(frame_count/num_frames))
#        frames = []
#        success = True
#        for frame_number in range(num_frames):
#            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, interval*frame_number)
            # Capture frame-by-frame
#            ret, frame = cap.read()
#            if ret:
#                # Crop the current frame
#                height = frame.shape[0]
#                width = frame.shape[1]
#                vert_indent = int(math.ceil((height - crop_dim)/2))
#                horiz_indent = int(math.ceil((width - crop_dim)/2))
#                frame = frame[vert_indent:vert_indent+crop_dim, 
#                              horiz_indent:horiz_indent+crop_dim, :]  
#                frame = frame.astype('float32') / 256   
#                frames.append(frame)
#            else:
#                print("Problem reading frame from file " + filenames[j])
#                success = False
                
                # When everything done, release the capture
#        cap.release()
#        if success:
#batch_labels.append(classes[filenames[j]])
#batch_frames.append(frames)
            
    np_batch_frames = np.asarray(batch_frames)
    if mean_img is not None:
        np_batch_frames -= mean_img
    np_batch_labels = np.asarray(batch_labels)
    actual_batch_size = len(batch_frames)
    # print(np_batch_frames.shape)
    # print(np_batch_labels)
#    print('files read')
    return np_batch_frames, np_batch_labels, actual_batch_size

def run_model(session, predict, loss_val, filenames, classes, number_to_class,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False, crop_dim=270, num_frames=10, mean_img=None):
    # have tensorflow compute accuracy
    predicted_class = tf.argmax(predict,1)
    correct_prediction = tf.equal(predicted_class, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indices
    train_indices = np.arange(len(filenames))
    np.random.shuffle(train_indices)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training


    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []

        # print("==================")
        # print("Printing weights")
        # print("==================")
        # for v in tf.trainable_variables():
        #     print(v)
        #     var = session.run(v)
        #     print(var)
        # print("=============")
        # print("done printing weights")


        # make sure we iterate over the dataset once
        #print(len(filenames))
        #print(int(math.ceil(len(filenames)/batch_size)))
        for i in range(int(math.ceil(len(filenames)/batch_size))):
            np_batch_frames, np_batch_labels, actual_batch_size = (
                get_batch_frames(filenames, batch_size, train_indices,
                                 number_to_class, classes, i, num_frames, crop_dim, mean_img=mean_img))

            # create a feed dictionary for this batch
            feed_dict = {X: np_batch_frames,
                         y: np_batch_labels,
                         is_training: training_now }

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step

            #loss, corr, ret_optimizer = session.run(variables,feed_dict=feed_dict)
            print('session running')
            if training_now:
                loss, corr, ret_optimizer, y_pred, class_pred = session.run([mean_loss,correct_prediction,training, predict, predicted_class],feed_dict=feed_dict)
            else:
                loss, corr, acc, y_pred, class_pred = session.run([mean_loss,correct_prediction,accuracy, predict, predicted_class],feed_dict=feed_dict)
            print('session done running')
            #print('y_pred', y_pred)
#            print('real labels', np_batch_labels)

 #           print('predicted class', class_pred)
            #print(ret_optimizer)
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
        #     # print every now and then
        #if training_now and (iter_cnt % print_every) == 0:
            print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                  .format(iter_cnt,loss,np.sum(corr)/float(actual_batch_size)))
            iter_cnt += 1
        total_correct = correct/float(len(filenames))
        total_loss = np.sum(losses)/len(filenames)
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
            # if plot_losses:
            #     plt.plot(losses)
            #     plt.grid(True)
            #     plt.title('Epoch {} Loss'.format(e+1))
            #     plt.xlabel('minibatch number')
            #     plt.ylabel('minibatch loss')
            #     plt.show()
    return total_loss,total_correct

def compute_mean_img(filenames, crop_dim, classes, number_to_class, num_frames):
    if os.path.exists('mean_img.npy'):
        return np.load('mean_img.npy')
    else:
        batch_size = 64
        mean_imgs = np.zeros((int(math.ceil(len(filenames)/batch_size)), crop_dim, crop_dim, 3))
        print('computing mean image')
        for i in range(int(math.ceil(len(filenames)/batch_size))):
            np_batch_frames, np_batch_labels, actual_batch_size = (
                get_batch_frames(filenames, batch_size, np.arange(len(filenames)),
                                 number_to_class, classes, i, num_frames, crop_dim))
            mean_imgs[i] = np.mean(np_batch_frames, axis=(0,1))
        mean_img = np.mean(mean_imgs, axis=0)
        print('finished computing mean image')
        np.save('mean_img.npy', mean_img)
        return mean_img

#def main():
classes, class_to_number, number_to_class, filenames_train, filenames_val, filenames_test = read_csv('SVW.csv', 3400, 300)
mean_img = compute_mean_img(filenames_train, 270, classes, number_to_class, 10)
X = tf.placeholder(tf.float32, [None, 10, 270, 270, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
y_pred, frames, before_relu = conv_model_3d.simple_model(X)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y)
mean_loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(1e-3)
train_step = optimizer.minimize(mean_loss)
with tf.Session() as sess:
    with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0"
        sess.run(tf.global_variables_initializer())
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
#        print(num_params)
        print('Training')
        for i in range(5):
            print('starting Epoch ', i+1)
            run_model(sess,y_pred,mean_loss,filenames_train,classes,
                      number_to_class,1,64,64,train_step,True, mean_img=mean_img)
            
            print('Validation')
            run_model(sess,y_pred,mean_loss,filenames_val,classes,
                      number_to_class,1,64, mean_img=mean_img)
        
        
#        session, predict, loss_val, filenames, classes, number_to_class,
#              epochs=1, batch_size=64, print_every=100,
#              training=None, plot_losses=False, crop_dim=270, num_frames=10
        
#    run_model(filenames, classes, number_to_class, 1, 1)

#if __name__ == "__main__":
#        main()
