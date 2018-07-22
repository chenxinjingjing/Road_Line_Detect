'''
code by kevin on 2017 09 23

Southeast University

kevin_wang435@163.com



'''




#--coding:utf-8--

import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray


train_dir='/home/nvidia/wxl_peter/tf/traffic_sign/data/train/'
logs_train_dir = '/home/nvidia/wxl_peter/tf/traffic_sign/logs0/train/'

INPUT_NODE=3072
LAYER1_NODE=100
OUTPUT_NODE=62

learning_rate = 0.001
MAX_STEP=401

def load_data(data_dir):
   
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))
    return images, labels


def display_images_and_labels(images, labels):
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()




def display_label_images(images, label):
    limit = 24  # show a max of 24 images
    plt.figure(figsize=(15, 5))
    i = 1
    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  # 3 rows, 8 per row
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()

def display_samples_images(sample_images,sample_labels,predicted):
    # Display the predictions and the ground truth visually.
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(sample_images)):
        truth = sample_labels[i]
        prediction = predicted[i]
        plt.subplot(5, 2,1+i)
        plt.axis('off')
        color='green' if truth == prediction else 'red'
        plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), fontsize=12, color=color)

        plt.imshow(sample_images[i])
    plt.show()



def inference(input_tensor,avg_class):
        with tf.name_scope('layer1'):
	    with tf.name_scope('weights1'):
                weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
		tf.summary.histogram('layer1/weights1', weights1)
            with tf.name_scope('biases1'):
                biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
		tf.summary.histogram('layer1/biases1', biases1)	
        with tf.name_scope('layer2'):
	    with tf.name_scope('weights2'):
                weights2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
                tf.summary.histogram('layer2/weights2', weights2)
	    with tf.name_scope('biases2'):
                biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
                tf.summary.histogram('layer2/biases2', biases2)
	if avg_class==None:
		layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)

		return tf.matmul(layer1,weights2)+biases2

	else:
		layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
		return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)


def inference0(images_flat):

	logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
	return logits

def losses(logits, labels):
  
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

def trainning(loss, learning_rate):

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op



def evaluation(logits, labels):

  with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy




########################################################33

images, labels = load_data(train_dir)

images32 = [skimage.transform.resize(image, (32, 32))
                for image in images]

display_images_and_labels(images32, labels)
display_label_images(images, 30)

labels_a = np.array(labels)
images_a = np.array(images32)
print("labels: ", labels_a.shape, "\nimages: ",images_a.shape)





# Create a graph to hold the model.
graph = tf.Graph()

# Create model in the graph.
with graph.as_default():
    
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3],name='x-input')
    labels_ph = tf.placeholder(tf.int32, [None],name='y-output')
    
    tf.summary.image('input',images_ph,10)
    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    
    images_flat = tf.contrib.layers.flatten(images_ph)


    # Fully connected layer. 
    # Generates logits of size [None, 62]
######### inference0
    logits = inference0(images_flat)
######### infenence 
     
	
    """
    weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

	
    weights2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    """ 
    #logits=inference(images_flat,None)
   
    

    loss = losses(logits,labels)
    train = trainning(loss, learning_rate)


    predicted_labels = tf.argmax(logits, 1)

    init=tf.global_variables_initializer()


    #print("images_flat: ", images_flat)
    #print("logits: ", logits)
    #print("loss: ", loss)
    #print("predicted_labels: ", predicted_labels)

    summary_op = tf.summary.merge_all()
    # Create a session to run the graph we created.
    session = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, session.graph)
    saver = tf.train.Saver()

   # First step is always to initialize all variables. 
   # We don't care about the return value, though. It's None.
    _ = session.run([init])
    

    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=session, coord=coord)
    
    #try:
    for step in np.arange(MAX_STEP):
    #        if coord.should_stop():break
            _, loss_value = session.run([train, loss], feed_dict={images_ph: images_a, labels_ph: labels_a})
            if (step == MAX_STEP):
                result = session.run(summary_op)
                train_writer.add_summary(result, step)

                
            else:

                    if step % 20 == 0:
                        print("Step %d, Loss=%.6f " % (step*10, loss_value))
                        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                        saver.save(session, checkpoint_path, global_step=step)
                
    #except tf.errors.OutOfRangeError:
    #    print('Done training -- epoch limit reached')
    #finally:
    #    coord.request_stop()
        
    #coord.join(threads)
    #session.close()

    #for i in range(MAX_STEP):
    #    _, loss_value = session.run([train, loss], 
     #                           feed_dict={images_ph: images_a, labels_ph: labels_a})
      #  if (i==MAX_STEP):
       #     result = session.run(summary_op) 
         #   train_writer.add_summary(result, i)
       # else:
        #    if i % 10 == 0:
         #       print("Loss: ", loss_value)
   	      
   
#Pick 10 random images
sample_indexes = random.sample(range(len(images32)), 10)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: sample_images})[0]
print(sample_labels)
print(predicted)

display_samples_images(sample_images,sample_labels,predicted)







