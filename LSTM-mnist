import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.rnn as rnn
tf.reset_default_graph()
mnist=input_data.read_data_sets('./MNIST_data/', one_hot=True)
test_x = mnist.test.images
test_y = mnist.test.labels
#Hyper Parameters
learning_rate=0.01
n_steps=28
n_inputs=28
n_hiddens=64
n_layers=2
n_classes=10
#tensor placeholder
with tf.name_scope('inputs'):
    x=tf.placeholder(tf.float32,[None,n_inputs*n_steps],name='x_input')
    y=tf.placeholder(tf.float32,[None,n_classes],name='y_input')
    keep_prob=tf.placeholder(tf.float32,name='keep_prob')
    batch_size=tf.placeholder(tf.int32,[],name='batch_size')
#weight and bias
with tf.name_scope('weight'):
    Weights=tf.Variable(tf.truncated_normal([n_hiddens,n_classes],stddev=0.1),dtype=tf.float32,name='W')
    tf.summary.histogram('outlayer_weights',Weights)
with tf.name_scope('biases'):
    Biases=tf.Variable(tf.random_normal([n_classes],name='bias'),name='b')
    tf.summary.histogram('outlayer_biases',Biases)
#RNN structure
def RNN_LSTM(x,weights,biases):
    x=tf.reshape(x,[-1,n_steps,n_inputs])
    def attn_cell():
        lstm_cell = rnn.BasicLSTMCell(n_hiddens)
        with tf.name_scope('lstm_dropout'):
            return rnn.DropoutWrapper(lstm_cell,output_keep_prob=keep_prob)
    enc_cell=[]
    for i in range(0,n_layers):
        enc_cell.append(attn_cell())
    with tf.name_scope('lstm_cell_layers'):
        mlstm_cell=rnn.MultiRNNCell(enc_cell,state_is_tuple=True)
    _init_state=mlstm_cell.zero_state(batch_size,dtype=tf.float32)
    out_puts,states=tf.nn.dynamic_rnn(mlstm_cell,x,initial_state=_init_state,dtype=tf.float32,time_major=False)
    return tf.nn.softmax(tf.matmul(out_puts[:,-1,:],weights)+biases)
with tf.name_scope('out_layer'):
    pred=RNN_LSTM(x,weights=Weights,biases=Biases)
    tf.summary.histogram('outputs',pred)
with tf.name_scope('loss'):
    cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=[1]))
    tf.summary.histogram('loss',cost)
with tf.name_scope('trian'):
    train_op=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
with tf.name_scope('accuracy'):
    accuracy=tf.metrics.accuracy(labels=tf.argmax(y,axis=1),predictions=tf.argmax(pred,axis=1))[1]
    tf.summary.scalar('accuracy',accuracy)
merged=tf.summary.merge_all()
init=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init)
    train_writer=tf.summary.FileWriter('./log/train',sess.graph)
    test_writer=tf.summary.FileWriter('./log/test',sess.graph)
    step=1
    for i in range(2000):
        _batch_size=128
        batch_x,batch_y=mnist.train.next_batch(_batch_size)

        sess.run(train_op,feed_dict={x:batch_x,y:batch_y,keep_prob:1.0,batch_size:_batch_size})
        if (i+1)%100==0:
            train_result=sess.run(merged,feed_dict={x:batch_x,y:batch_y,keep_prob:1.0,batch_size:_batch_size})
            test_result=sess.run(merged,feed_dict={x:test_x,y:test_y,keep_prob:1.0,batch_size:test_x.shape[0]})
            train_writer.add_summary(train_result,i+1)
            test_writer.add_summary(test_result,i+1)
    print('Optimization Finished')
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x:test_x, y:test_y, keep_prob:1.0, batch_size:test_x.shape[0]}))
