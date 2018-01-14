
import os
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle

mode = "train"
 
char_size = 3755
Epochs = 1
Batch_size = 300
learning_rate = 0
checkpoint_dir = 'C:\grow\AI\data\checkpoint'
train_data_dir = 'C:\grow\AI\data\HWDB1.1tst_gnt'
test_data_dir = 'C:\grow\AI\data\HWDB1.1trn_gnt'
data_dir = 'C:\grow\AI\data\'
word_dict='C:\grow\AI\data\char_dict'
debug = True


def LogMessage(log_messsage):
    if debug:
        print(log_messsage)
def LogMessageList(log_messsageList):
    if debug:
        print("Size of list:" ,len(log_messsageList), ". print top 10:")
        for log_messsage in log_messsageList[:10]:
            print(log_messsage)

#PNG Read section
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

#Gnt read section	
def read_from_gnt_dir(gnt_dir=train_data_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break
            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
            tagcode = header[5] + (header[4]<<8)
            width = header[6] + (header[7]<<8)
            height = header[8] + (header[9]<<8)
            if header_size + width*height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
            yield image, tagcode
    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, tagcode in one_file(f):
                    yield image, tagcode
	
def batch_data(file_labels,sess, batch_size=128):
    image_list = [file_label[0] for file_label in file_labels]
    label_list = [int(file_label[1]) for file_label in file_labels]
    print 'tag2 {0}'.format(len(image_list))
    images_tensor = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels_tensor = tf.convert_to_tensor(label_list, dtype=tf.int64)
    input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor])
 
    labels = input_queue[1]
    images_content = tf.read_file(input_queue[0])
    # images = tf.image.decode_png(images_content, channels=1)
    images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
    # images = images / 256
    images =  pre_process(images)
    # print images.get_shape()
    # one hot
    labels = tf.one_hot(labels, 3755)
    image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,min_after_dequeue=10000)
    # print 'image_batch', image_batch.get_shape()
 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return image_batch, label_batch, coord, threads
	
	
char_set = set()
for _, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    char_set.add(tagcode_unicode)
char_list = list(char_set)
char_dict = dict(zip(sorted(char_list), range(len(char_list))))
print len(char_dict)

f = open(word_dict, 'wb')
pickle.dump(char_dict, f)
f.close()
train_counter = 0
test_counter = 0



for image, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    im = Image.fromarray(image)
    dir_name = '../data/train/' + '%0.5d'%char_dict[tagcode_unicode]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    im.convert('RGB').save(dir_name+'/' + str(train_counter) + '.png')
    train_counter += 1
for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    im = Image.fromarray(image)
    dir_name = '../data/test/' + '%0.5d'%char_dict[tagcode_unicode]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    im.convert('RGB').save(dir_name+'/' + str(test_counter) + '.png')
    test_counter += 1
	
def batch_data(file_labels,sess, batch_size=128):
    image_list = [file_label[0] for file_label in file_labels]
    label_list = [int(file_label[1]) for file_label in file_labels]
    print 'tag2 {0}'.format(len(image_list))
    images_tensor = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels_tensor = tf.convert_to_tensor(label_list, dtype=tf.int64)
    input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor])
 
    labels = input_queue[1]
    images_content = tf.read_file(input_queue[0])
    # images = tf.image.decode_png(images_content, channels=1)
    images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
    # images = images / 256
    images =  pre_process(images)
    # print images.get_shape()
    # one hot
    labels = tf.one_hot(labels, 3755)
    image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,min_after_dequeue=10000)
    # print 'image_batch', image_batch.get_shape()
 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return image_batch, label_batch, coord, threads
 
def pre_process(images):
    if FLAGS.random_flip_up_down:
        images = tf.image.random_flip_up_down(images)
    if FLAGS.random_flip_left_right:
        images = tf.image.random_flip_left_right(images)
    if FLAGS.random_brightness:
        images = tf.image.random_brightness(images, max_delta=0.3)
    if FLAGS.random_contrast:
        images = tf.image.random_contrast(images, 0.8, 1.2)
    new_size = tf.constant([FLAGS.image_size,FLAGS.image_size], dtype=tf.int32)
    images = tf.image.resize_images(images, new_size)
    return images
	
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

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

def model(images, labels):
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
    #flatten = slim.flatten(pooling)
    #fc = slim.fully_connected(flatten, 1024, activation_fn=tf.nn.tanh, scope='fc')
    W_fc1 = weight_variable([8 * 8 * 256, 8192])
    b_fc1 = bias_variable([8192])
    # [n_samples, 8, 8, 256] ->> [n_samples, 8*8*256]
    h_pool2_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

    # func2 layer ##
    #logits = slim.fully_connected(fc, char_size, activation_fn=None, scope='output_logit')
    W_fc2 = weight_variable([8192, char_size])
    b_fc2 = bias_variable([char_size])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # the error between prediction and real data
    # loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    #loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(prediction), reduction_indices=[1]))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), labels), tf.float32))

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)

    #train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)
    probabilities = tf.nn.softmax(prediction)
    pred = tf.identity(probabilities, name='prediction')

    return loss, accuracy, global_step, train_op, pred

def modelSlim(images, labels=None):
    endpoints = {}
    conv_1 = slim.conv2d(images, 32, [3,3],1, padding='SAME')
    max_pool_1 = slim.max_pool2d(conv_1, [2,2],[2,2], padding='SAME')
    conv_2 = slim.conv2d(max_pool_1, 64, [3,3],padding='SAME')
    max_pool_2 = slim.max_pool2d(conv_2, [2,2],[2,2], padding='SAME')
    flatten = slim.flatten(max_pool_2)
    out = slim.fully_connected(flatten,3755, activation_fn=None)
    global_step = tf.Variable(initial_value=0)
    if labels is not None:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, labels))
        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, global_step=global_step)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(labels, 1)), tf.float32))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()
    output_score = tf.nn.softmax(out)
    predict_val_top3, predict_index_top3 = tf.nn.top_k(output_score, k=3)

    endpoints['global_step'] = global_step
    if labels is not None:
        endpoints['labels'] = labels
        endpoints['train_op'] = train_op
        endpoints['loss'] = loss
        endpoints['accuracy'] = accuracy
        endpoints['merged_summary_op'] = merged_summary_op
    endpoints['output_score'] = output_score
    endpoints['predict_val_top3'] = predict_val_top3
    endpoints['predict_index_top3'] = predict_index_top3
    return endpoints

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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print ("== start training ==")
    with tf.Session(config = config) as sess:
        loss, accuracy, global_step, train_op, pred = model(image_x, label_y)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        saver = tf.train.Saver()
        epoch = 0
        while epoch in range(Epochs):
            for steps in range(20000):
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
		
def trainSlim():
    sess = tf.Session()
    file_labels = get_imagesfile(train_data_dir)
    images, labels, coord, threads = batch_data(file_labels, sess)
    endpoints = network(images, labels)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.train.SummaryWriter('./log' + '/train',sess.graph)
    test_writer = tf.train.SummaryWriter('./log' + '/val')
    start_step = 0
    if FLAGS.restore:
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print "restore from the checkpoint {0}".format(ckpt)
            start_step += int(ckpt.split('-')[-1])
    logger.info(':::Training Start:::')
    try:
        while not coord.should_stop():
        # logger.info('step {0} start'.format(i))
            start_time = time.time()
            _, loss_val, train_summary, step = sess.run([endpoints['train_op'], endpoints['loss'], endpoints['merged_summary_op'], endpoints['global_step']])
            train_writer.add_summary(train_summary, step)
            end_time = time.time()
            logger.info("the step {0} takes {1} loss {2}".format(step, end_time-start_time, loss_val))
            if step > FLAGS.max_steps:
                break
            # logger.info("the step {0} takes {1} loss {2}".format(i, end_time-start_time, loss_val))
            if step % FLAGS.eval_steps == 1:
                accuracy_val,test_summary, step = sess.run([endpoints['accuracy'], endpoints['merged_summary_op'], endpoints['global_step']])
                test_writer.add_summary(test_summary, step)
                logger.info('===============Eval a batch in Train data=======================')
                logger.info( 'the step {0} accuracy {1}'.format(step, accuracy_val))
                logger.info('===============Eval a batch in Train data=======================')
            if step % FLAGS.save_steps == 1:
                logger.info('Save the ckpt of {0}'.format(step))
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=endpoints['global_step'])
    except tf.errors.OutOfRangeError:
        # print "============train finished========="
        logger.info('==================Train Finished================')
        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=endpoints['global_step'])
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


def validation():
    # it should be fixed by using placeholder with epoch num in train stage
    sess = tf.Session()
 
    file_labels = get_imagesfile(FLAGS.test_data_dir)
    test_size = len(file_labels)
    print test_size
    val_batch_size = FLAGS.val_batch_size
    test_steps = test_size / val_batch_size
    print test_steps
    # images, labels, coord, threads= batch_data(file_labels, sess)
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
    labels = tf.placeholder(dtype=tf.int32, shape=[None,3755])
    # read batch images from file_labels
    # images_batch = np.zeros([128,64,64,1])
    # labels_batch = np.zeros([128,3755])
    # labels_batch[0][20] = 1
    #
    endpoints = network(images, labels)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
        # logger.info("restore from the checkpoint {0}".format(ckpt))
    # logger.info('Start validation')
    final_predict_val = []
    final_predict_index = []
    groundtruth = []
    for i in range(test_steps):
        start = i* val_batch_size
        end = (i+1)*val_batch_size
        images_batch = []
        labels_batch = []
        labels_max_batch = []
        logger.info('=======start validation on {0}/{1} batch========='.format(i, test_steps))
        for j in range(start,end):
            image_path = file_labels[j][0]
            temp_image = Image.open(image_path).convert('L')
            temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size),Image.ANTIALIAS)
            temp_label = np.zeros([3755])
            label = int(file_labels[j][1])
            # print label
            temp_label[label] = 1
            # print "====",np.asarray(temp_image).shape
            labels_batch.append(temp_label)
            # print "====",np.asarray(temp_image).shape
            images_batch.append(np.asarray(temp_image)/255.0)
            labels_max_batch.append(label)
        # print images_batch
        images_batch = np.array(images_batch).reshape([-1, 64, 64, 1])
        labels_batch = np.array(labels_batch)
        batch_predict_val, batch_predict_index = sess.run([endpoints['predict_val_top3'],
                        endpoints['predict_index_top3']], feed_dict={images:images_batch, labels:labels_batch})
        logger.info('=======validation on {0}/{1} batch end========='.format(i, test_steps))
        final_predict_val += batch_predict_val.tolist()
        final_predict_index += batch_predict_index.tolist()
        groundtruth += labels_max_batch
    sess.close()
    return final_predict_val, final_predict_index, groundtruth
	
def inference(image):
    temp_image = Image.open(image).convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size),Image.ANTIALIAS)
    sess = tf.Session()
    logger.info('========start inference============')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
    endpoints = network(images)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
    predict_val, predict_index = sess.run([endpoints['predict_val_top3'],endpoints['predict_index_top3']], feed_dict={images:temp_image})
    sess.close()
    return final_predict_val, final_predict_index
	
def main():
    trainSlim()

if __name__ == '__main__':
    main()

