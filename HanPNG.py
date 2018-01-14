import os
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

mode = "train"
 
char_size = 3755
Epochs = 1
Batch_size = 300
learning_rate = 0
checkpoint_dir = '/aiml/dfs/checkpoint/'
train_data_dir = '/aiml/data/train/'
test_data_dir = '/aiml/data/test/'
data_dir = "/aiml/data/"
debug = True


def LogMessage(log_messsage):
    if debug:
        print(log_messsage)
def LogMessageList(log_messsageList):
    if debug:
        print("Size of list:" ,len(log_messsageList), ". print top 10:")
        for log_messsage in log_messsageList[:10]:
            print(log_messsage)

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
    LogMessageList(image_names)
    LogMessageList(label_list)
    return image_names, label_list
 
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
    images = read_convert_image(input_queue[0])
    labels = input_queue[1]
    
    #collect batches
    image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=Batch_size, capacity=50000, min_after_dequeue=10000)
                                                                                                                                                            
    print ("created input pipeline")
    return image_batch, label_batch

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial  = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    #stride[1, x_movement, y_movement, 1]
    #Must have strides[0] =  strdes[4] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    #stride[1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides =  [1, 2, 2,1], padding='SAME')


def model(images, labels, keep_prob):
	
    # conv 1 layer ## input 64 * 64
    #conv = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv')
    #pooling = slim.max_pool2d(conv, [2, 2], [2, 2], padding='SAME')
    W_conv1 = weight_variable([5, 5, 1, 64])  # patch 5x5, in size 1, out size 64
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)  # output size 64x64x64
    h_pool1 = max_pool_2x2(h_conv1)  # output size 32* 32 *64

    # conv2 layer ##
    W_conv2 = weight_variable([5, 5, 64, 128])  # patch 5*5, in size 64, out size 128
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 32*32*128
    h_pool2 = max_pool_2x2(h_conv2)  # output size 16* 16 *128

    # conv3 layer ##
    W_conv3 = weight_variable([5, 5, 128, 256])  # patch 5*5, in size 128, out size 256
    b_conv3 = bias_variable([256])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)  # output size 16*16*256
    h_pool3 = max_pool_2x2(h_conv3)  # output size 8* 8 *256

     # func1 layer ##
    #flatten = slim.flatten(h_pool3)
    #fc = slim.fully_connected(flatten, 1024, activation_fn=tf.nn.tanh, scope='fc')
    W_fc1 = weight_variable([8 * 8 * 256, 1024])
    b_fc1 = bias_variable([1024])
    # [n_samples, 8, 8, 256] ->> [n_samples, 8*8*256]
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # func2 layer ##
    #logits = slim.fully_connected(fc, char_size, activation_fn=None, scope='output_logit')
    W_fc2 = weight_variable([1024, char_size])
    b_fc2 = bias_variable([char_size])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # the error between prediction and real data
    # loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    #loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(prediction), reduction_indices=[1]))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), labels), tf.float32))

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)

    #train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
	
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)
    probabilities = tf.nn.softmax(prediction)
    pred = tf.identity(probabilities, name='prediction')
	
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
	
    return loss, accuracy, global_step, train_op, pred

def modelDemo(images, labels):
    # with tf.device('/cpu:0'):
    conv = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv')
    pooling = slim.max_pool2d(conv, [2, 2], [2, 2], padding='SAME')

    flatten = slim.flatten(pooling)
    fc = slim.fully_connected(flatten, 1024, activation_fn=tf.nn.tanh, scope='fc')
    logits = slim.fully_connected(fc, char_size, activation_fn=None, scope='output_logit')
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step = global_step)
    probabilities = tf.nn.softmax(logits)
    pred = tf.identity(probabilities, name = 'prediction') 
    
    return loss, accuracy, global_step, train_op, pred
    
def train():
    train_image_names, train_label_list = get_input_lists(train_data_dir)
    test_image_names, test_label_list = get_input_lists(test_data_dir)

    train_data_x, train_data_y = get_batches(train_image_names, train_label_list)    
    test_data_x, test_data_y = get_batches(test_image_names, test_label_list)

    image_x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='input_image')
    label_y = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
	
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print ("== start training ==")
    with tf.Session(config = config) as sess:
        loss, accuracy, global_step, train_op, pred = model(image_x, label_y, keep_prob)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        saver = tf.train.Saver()
        epoch = 0
        while epoch in range(Epochs):
            for steps in range(895034):
                train_x_batch, train_y_batch = sess.run([train_data_x, train_data_y])
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={image_x: train_x_batch, label_y: train_y_batch, keep_prob:0.5})
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


def main():
    train()

if __name__ == '__main__':
    main()

