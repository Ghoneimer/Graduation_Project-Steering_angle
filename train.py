import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import prepare
import model

LOGDIR = './save/modelsimple_2'

sess = tf.InteractiveSession()

L2NormConst = 0.001

train_vars = tf.trainable_variables()

#loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y[0])))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(model.y_, tf.reshape(model.y,[-1]))))) 
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()
'''
saver.restore(sess, "save/model2/modelsteering")
'''

# create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

epochs = 30
batch_size = 32
num_of_batches=int((prepare.num_images*2)/batch_size)
cost_ref=1

# train over the dataset about 30 times
for epoch in range(epochs):
  cost=0
  for i in range(num_of_batches):
    xs, ys = prepare.batch_gen(prepare.X_train, batch_size)
    train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
    loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    cost+=loss_value/num_of_batches
    '''
    #loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    print(loss_value)
    print(len(loss_value))
    '''
    if i % 10 == 0:
      xs, ys = prepare.batch_gen(prepare.X_validation, batch_size)
      loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    # write logs at every iteration
    summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * (prepare.num_images*2)/batch_size + i)

    #if i % batch_size == 0:
  if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
  if (cost < cost_ref):
    checkpoint_path = os.path.join(LOGDIR, "modelsteering")
    filename = saver.save(sess, checkpoint_path)
    cost_ref=cost
    print("Model saved in file: %s" % filename)
  print("Cost after epoch %i = %f and cost reference = %f" %(epoch,cost,cost_ref))
