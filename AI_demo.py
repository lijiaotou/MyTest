import os
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle

#mode = "train" / test / inference"
mode = "inference"
char_size = 3755
#char_size = 31
Epochs = 1
Batch_size = 100

checkpoint_dir = '/aiml/dfs/checkpoint/'
#Totoal training set : 895034
train_data_dir = '/aiml/data/train/'
test_data_dir = '/aiml/data/test/'
test_png_dir = "/aiml/data2/"
data_dir = "/aiml/data/"
word_dict_location = "/aiml/data/word_dict"
resultTxT_location = "/aiml/result/result.txt"

#word_dict读取：
f_word_dict = open(word_dict_location, 'rb')
Word_dict = pickle.load(f_word_dict)
f_word_dict.close()

def printWordbyID(idArray):
    result = ""
    for id in idArray:
        result += Word_dict[str(id)]
    print("Result is:", result)
    fileResult =open(resultTxT_location,'w')
    fileResult.write(result)
    fileResult.close()

def get_input_lists(data_dir):
    image_names = []
    label_list = []
    label_len = []
    for root, sub_folder, file_list in os.walk(data_dir):
        image_names += [os.path.join(root, file_path) for file_path in file_list]

        for name in sub_folder:
            current_dir = os.path.join(root, name)
            set_images = os.listdir(current_dir)
            label_len.append(len(set_images))
    index = 0
    while index < (len(label_len)):
        i = 0
        while i in range(label_len[index]):    
            label_list.append(index)
            i += 1
        index += 1
    return image_names, label_list

def get_image_lists(data_dir):
    image_names = []
    for root, sub_folder, file_list in os.walk(data_dir):
        image_names += [os.path.join(root, file_path) for file_path in file_list]

    return image_names
 
def read_convert_image(image_name):
    images_content = tf.read_file(image_name)
    image = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
    new_size = tf.constant([64, 64], dtype=tf.int32)
    new_image = tf.image.resize_images(image, new_size)
    return new_image 

def get_batches(image_names, label_list):

    #convert strings into tensors
    all_images = tf.convert_to_tensor(image_names, dtype = tf.string)
    all_labels = tf.convert_to_tensor(label_list, dtype = tf.int64)
    #create input queue
    input_queue = tf.train.slice_input_producer([all_images, all_labels], num_epochs=None)
    #process path into image and label
    labels = input_queue[1]
    images = read_convert_image(input_queue[0]) 
    #collect batches
    image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=Batch_size, capacity=5000, min_after_dequeue=1000)
                                                                                                                                                            
    print ("created input pipeline")
    return image_batch, label_batch

def get_Normalbatches(image_names):

    #convert strings into tensors
    label_list = np.ones_like(image_names)
    all_images = tf.convert_to_tensor(image_names, dtype = tf.string)
    all_labels = tf.convert_to_tensor(label_list, dtype = tf.int64)

    #create input queue
    input_queue = tf.train.slice_input_producer([all_images, all_labels], shuffle=False, num_epochs=None)
    #process path into image and label
    labels = input_queue[1]
    images = read_convert_image(input_queue[0]) 
    #collect batches
    image_batch, label_batch = tf.train.batch([images, labels], batch_size=Batch_size, capacity=5000)
                                                                                                                                                            
    print ("created input pipeline whit shuffle=False")
    return image_batch, label_batch
    
def model(images, labels):        
    # with tf.device('/cpu:0'):
    conv = slim.conv2d(images, 64, [5, 5], 1, padding='SAME', scope='conv')
    pooling = slim.max_pool2d(conv, [2, 2], [2, 2], padding='SAME')

    conv2 = slim.conv2d(pooling, 128, [3, 3], 1, padding='SAME', scope='conv2')
    pooling2 = slim.max_pool2d(conv2, [2, 2], [2, 2], padding='SAME')

    conv3 = slim.conv2d(pooling2, 256, [3, 3], 1, padding='SAME', scope='conv3')
    pooling3 = slim.max_pool2d(conv3, [2, 2], [2, 2], padding='SAME')

    conv4 = slim.conv2d(pooling3, 512, [3, 3], 1, padding='SAME', scope='conv4')
    pooling4 = slim.max_pool2d(conv4, [2, 2], [2, 2], padding='SAME')

    flatten = slim.flatten(pooling4)
    fc = slim.fully_connected(flatten, 1024, activation_fn=tf.nn.tanh, scope='fc')
    logits = slim.fully_connected(fc, char_size, activation_fn=None, scope='output_logit')
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

    number = tf.argmax(logits, 1)

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step = global_step)

    #rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)
    #train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)
    
    probabilities = tf.nn.softmax(logits)
    pred = tf.identity(probabilities, name = 'prediction') 
    
    return loss, accuracy, global_step, train_op, pred, number
    
def train():
    train_image_names, train_label_list = get_input_lists(train_data_dir)
    test_image_names, test_label_list = get_input_lists(test_data_dir)
    train_data_x, train_data_y = get_batches(train_image_names, train_label_list)    
    test_data_x, test_data_y = get_batches(test_image_names, test_label_list)

    image_x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='input_image')
    label_y = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print ("== start training ==")
    with tf.Session(config = config) as sess:
        loss, accuracy, global_step, train_op, pred, number = model(image_x, label_y)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  
        if ckpt and ckpt.model_checkpoint_path:  
           saver.restore(sess, ckpt.model_checkpoint_path)
        epoch = 0
        while epoch in range(Epochs):
            for steps in range(895034):
                train_x_batch, train_y_batch = sess.run([train_data_x, train_data_y])
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={image_x: train_x_batch, label_y: train_y_batch})    
                if step % 10 == 0:
                    print ("step :", step, "loss is:", loss_value)
                
                if step % 100 == 0:
                    test_x_batch, test_y_batch = sess.run([test_data_x, test_data_y])
                    accuracy_value= sess.run([accuracy], feed_dict={image_x: test_x_batch, label_y: test_y_batch})
                    print ("in step ", step, "current accuracy is", accuracy_value)
                    saver.save(sess, checkpoint_dir +"new_model.ckpt", global_step=global_step)
                    
        coord.request_stop()  
        coord.join(threads)  
        sess.close()

def test():
    test_image_names, test_label_list = get_input_lists(test_data_dir)
    test_data_x, test_data_y = get_batches(test_image_names, test_label_list)

    image_x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='input_image')
    label_y = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
   
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print ("== start testing ==")
    with tf.Session(config = config) as sess:
        loss, accuracy, global_step, train_op, pred, number = model(image_x, label_y)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  
        if ckpt and ckpt.model_checkpoint_path:  
           saver.restore(sess, ckpt.model_checkpoint_path)
        test_x_batch, test_y_batch = sess.run([test_data_x, test_data_y])
        accuracy_value, number_value = sess.run([accuracy, number], feed_dict={image_x: test_x_batch, label_y: test_y_batch})
        print ("current accuracy is", accuracy_value)
        sess.close()
        
def inference():
    test_image_names = get_image_lists(test_png_dir)
    test_data_x, test_data_y = get_Normalbatches(test_image_names)
 

    image_x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='input_image')
    label_y = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
   
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print ("== start testing ==")
    with tf.Session(config = config) as sess:
        loss, accuracy, global_step, train_op, pred, number = model(image_x, label_y)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  
        if ckpt and ckpt.model_checkpoint_path:  
           saver.restore(sess, ckpt.model_checkpoint_path)
        test_x_batch, test_y_batch = sess.run([test_data_x, test_data_y])
        accuracy_value, number_value = sess.run([accuracy, number], feed_dict={image_x: test_x_batch, label_y: test_y_batch})
        printWordbyID(number_value)
        sess.close()
	
def main():
    if mode == "train":
        train()
    if mode == "test":
        test()
    if mode == "inference":
        inference()
        
if __name__ == '__main__':
    main()
