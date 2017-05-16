import cv2
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

import csv

def read_csv(filename):
    classes = {}
    class_to_number = {}
    number_to_class = {}
    unique_genres = set()
    filenames = []
    
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
            
    with open(filename, 'rU') as csvfile:
        vidreader = csv.DictReader(csvfile)
        for row in vidreader:
            classes[row['FileName']] = class_to_number[row['Genre']]
            
    return classes, class_to_number, number_to_class, filenames
    
def get_batch_frames(filenames, batch_size, train_indices,
                     number_to_class, classes, batch_num, num_frames, crop_dim):
    # generate indices for the batch
    start_idx = (batch_num*batch_size)%len(filenames)
    idx = train_indices[start_idx:start_idx+batch_size]
    #per-batch vars
    batch_frames = []
    batch_labels = []
    
    for j in idx:
        filepath = ('SVW/Videos/' + number_to_class[classes[filenames[j]]]
                    + '/' + filenames[j])
        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        print(filepath)
        frame_count = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        print(frame_count)
        interval = int(math.floor(frame_count/num_frames))
        frames = []
        success = True
        for frame_number in range(num_frames):
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, interval*frame_number)
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Crop the current frame
                height = frame.shape[0]
                width = frame.shape[1]
                vert_indent = int(math.ceil((height - crop_dim)/2))
                horiz_indent = int(math.ceil((width - crop_dim)/2))
                frame = frame[vert_indent:vert_indent+crop_dim, 
                              horiz_indent:horiz_indent+crop_dim, :]
                frames.append(frame)
            else:
                print("Problem reading frame from file " + filenames[j])
                success = False
                
                # When everything done, release the capture
        cap.release()
        if success:
            batch_labels.append(classes[filenames[j]])
            batch_frames.append(frames)
            
        np_batch_frames = np.asarray(batch_frames)
        np_batch_labels = np.asarray(batch_labels)
        actual_batch_size = len(batch_frames)
        print(np_batch_frames.shape)
        return np_batch_frames, np_batch_labels, actual_batch_size

#def run_model(session, predict, loss_val, filenames, classes,
#              number_to_class, epochs=1, batch_size=64, print_every=100,
#              training=None, plot_losses=False, crop_dim=270, num_frames=10):

def run_model(filenames, classes, number_to_class,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False, crop_dim=270, num_frames=10):
    # have tensorflow compute accuracy
#    correct_prediction = tf.equal(tf.argmax(predict,1), y)
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indices
    train_indices = np.arange(len(filenames))
    np.random.shuffle(train_indices)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
#    variables = [mean_loss,correct_prediction,accuracy]
#    if training_now:
#        variables[-1] = training

    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []

        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(len(filenames)/batch_size))):
            np_batch_frames, np_batch_labels, actual_batch_size = (
                get_batch_frames(filenames, batch_size, train_indices,
                                 number_to_class, classes, i, num_frames, crop_dim))

            # create a feed dictionary for this batch
#            feed_dict = {X: np_batch_frames,
#                         y: np_batch_labels),
#                         is_training: training_now }
            # get batch size
            print(np_batch_frames.shape)
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
        #     loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
        #     # aggregate performance stats
        #     losses.append(loss*actual_batch_size)
        #     correct += np.sum(corr)
            
        #     # print every now and then
        #     if training_now and (iter_cnt % print_every) == 0:
        #         print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
        #               .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
        #     iter_cnt += 1
        # total_correct = correct/filenames.shape[0]
        # total_loss = np.sum(losses)/filenames.shape[0]
        # print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
        #       .format(total_loss,total_correct,e+1))
        # if plot_losses:
        #     plt.plot(losses)
        #     plt.grid(True)
        #     plt.title('Epoch {} Loss'.format(e+1))
        #     plt.xlabel('minibatch number')
        #     plt.ylabel('minibatch loss')
        #     plt.show()
#    return total_loss,total_correct


def main():
    classes, class_to_number, number_to_class, filenames = read_csv('SVW/SVW_mini.csv')
    run_model(filenames, classes, number_to_class, 1, 1)

if __name__ == "__main__":
        main()
