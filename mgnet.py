import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type = str, required = True, help = "which dataset to use.  can choose between TFDS datasets such as mnist, cifar10, etc.")
parser.add_argument("--iterations", type = str, required = True, help = "number of iterations per mgnet block.  In the format 2,2,2,2 in which layer iterations are comma separated")
parser.add_argument("--u-channels", type = str, required = True, help = "number of channels for u convolution.  In the format 2,2,2,2 in which layer channels are comma separated")
parser.add_argument("--f-channels", type = str, required = True, help = "number of channels for f convolution.  In the format 2,2,2,2 in which layer channels are comma separated")
parser.add_argument("--batch-size", type = int, required = True, help = "number of samples for each batch")
parser.add_argument("--epochs", type = int, required = True, help = "maximum number of epochs during training")
parser.add_argument("--lr", type = float, required = True, help = "learning rate for Adam optimizer")
parser.add_argument("--graph", type = bool, required = True, help = "whether or not to generate loss/accuracy graph")
args = parser.parse_args()

dataset = args.dataset
iterations = args.iterations
u_channels = args.u_channels
f_channels = args.f_channels
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
graph = args.graph

iterations = [int(x) for x in iterations.split(",")]
u_channels = [int(x) for x in u_channels.split(",")]
f_channels = [int(x) for x in f_channels.split(",")]

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from datetime import datetime

(ds_train, ds_test), ds_info = tfds.load(
    dataset,
    split = ["train", "test"],
    as_supervised = True,
    with_info = True)

rescale = tf.keras.layers.Rescaling(1. / 255)
if dataset == "mnist":
  mean, variance = [.1307], np.square([.3081])
if dataset == "cifar10":
  mean, variance = [.4914, .4822, .4465], np.square([.2023, .1994, .2010])
if dataset == "cifar100":
  mean, variance = [.5071, .4865, .4409], np.square([.2673, .2564, .2762])
normalize = tf.keras.layers.Normalization(mean = mean,
                                          variance = variance)

def preprocess(ds, training):
  if training:
    layers = tf.keras.Sequential([
      rescale,
      tf.keras.layers.RandomTranslation(height_factor = .125,
                                        width_factor = .125,
                                        fill_mode = "constant"),
      tf.keras.layers.RandomFlip(mode = "horizontal"),
      normalize
    ])
    ds = ds.shuffle(ds_info.splits["train"].num_examples)
  else:
    layers = tf.keras.Sequential([rescale, normalize])

  ds = ds.batch(batch_size)
  ds = ds.map(lambda x, y: (layers(x), y),
              num_parallel_calls = tf.data.AUTOTUNE)
  ds = ds.cache()
  ds = ds.prefetch(tf.data.AUTOTUNE)

  return ds

ds_train = preprocess(ds_train, training = True)
ds_test = preprocess(ds_test, training = False)



class MgSmooth(tf.keras.layers.Layer):

  def __init__(self, iterations, u_channels, f_channels):
    super(MgSmooth, self).__init__()

    self.iterations = iterations
    self.A = tf.keras.layers.Conv2D(u_channels,
                                    (3, 3),
                                    strides = (1, 1),
                                    padding = "same",
                                    use_bias = False,
                                    kernel_initializer = "he_uniform")
    self.B = tf.keras.layers.Conv2D(f_channels,
                                    (3, 3),
                                    strides = (1, 1),
                                    padding = "same",
                                    use_bias = False,
                                    kernel_initializer = "he_uniform")

    self.A_bns, self.B_bns = [], []
    for _ in range(self.iterations):
      self.A_bns.append(tf.keras.layers.BatchNormalization())
      self.B_bns.append(tf.keras.layers.BatchNormalization())

  def call(self, u, f):
    for i in range(self.iterations):
      error = tf.nn.relu(self.A_bns[i](f - self.A(u)))
      u = u + tf.nn.relu(self.B_bns[i](self.B(error)))
    return u, f

class MgBlock(tf.keras.layers.Layer):

  def __init__(self, iterations, u_channels, f_channels, A_old):
    super(MgBlock, self).__init__()

    self.iterations = iterations
    self.Pi = tf.keras.layers.Conv2D(u_channels,
                                     (3, 3),
                                     strides = (2, 2),
                                     padding = "same",
                                     use_bias = False,
                                     kernel_initializer = "he_uniform")
    self.R = tf.keras.layers.Conv2D(f_channels,
                                    (3, 3),
                                    strides = (2, 2),
                                    padding = "same",
                                    use_bias = False,
                                    kernel_initializer = "he_uniform")
    self.A_old = A_old
    self.MgSmooth = MgSmooth(self.iterations, u_channels, f_channels)

    self.Pi_bn = tf.keras.layers.BatchNormalization()
    self.R_bn = tf.keras.layers.BatchNormalization()

  def call(self, u0, f0):
    u1 = tf.nn.relu(self.Pi_bn(self.Pi(u0)))
    error = tf.nn.relu(self.R_bn(self.R(f0 - self.A_old(u0))))
    f1 = error + self.MgSmooth.A(u1)
    u, f = self.MgSmooth(u1, f1)
    return u, f

class MgNet(tf.keras.Model):

  def __init__(self, iterations, u_channels, f_channels, in_shape, out_shape):
    super(MgNet, self).__init__()

    self.iterations = iterations
    self.in_shape = in_shape
    self.A_init = tf.keras.layers.Conv2D(u_channels[0],
                                         (3, 3),
                                         strides = (1, 1),
                                         padding = "same",
                                         use_bias = False,
                                         kernel_initializer = "he_uniform")
    self.A_bn = tf.keras.layers.BatchNormalization()

    self.blocks = []
    for i in range(len(self.iterations)):
      if i == 0:
        self.blocks.append(MgSmooth(iterations[i],
                                    u_channels[i],
                                    f_channels[i]))
        continue
      if i == 1:
        self.blocks.append(MgBlock(iterations[i],
                                   u_channels[i],
                                   f_channels[i],
                                   self.blocks[0].A))
        continue
      self.blocks.append(MgBlock(iterations[i],
                                 u_channels[i],
                                 f_channels[i],
                                 self.blocks[i - 1].MgSmooth.A))

    x = in_shape[0]
    for i in range(len(self.blocks) - 1):
      x = ((x + 2 - 3) // 2) + 1
    self.pool = tf.keras.layers.AveragePooling2D(pool_size = (x, x))
    self.fc = tf.keras.layers.Dense(out_shape,
                                    activation = "softmax")
  
  def call(self, u0):
    f = tf.nn.relu(self.A_bn(self.A_init(u0)))
    u = tf.multiply(f, 0)

    for block in self.blocks:
      u, f = block(u, f)
    u = self.pool(u)
    u = tf.squeeze(u, [-2, -3])
    u = self.fc(u)
    return u



tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices("GPU")
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
  model = MgNet(iterations,
                u_channels,
                f_channels,
                ds_info.features["image"].shape,
                ds_info.features["label"].num_classes)

  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  optimizer = tf.keras.optimizers.Adam(lr)
  model.compile(optimizer = optimizer, loss = loss, metrics = ["accuracy"])

  history = model.fit(ds_train,
                      epochs = epochs,
                      validation_data = ds_test)

model.summary()



if graph:
  loss = history.history["loss"]
  accuracy = history.history["accuracy"]
  val_loss = history.history["val_loss"]
  val_accuracy = history.history["val_accuracy"]
  timerange = range(len(loss))

  fig,ax = plt.subplots()
  train_loss_plot, = ax.plot(timerange, loss, color = "blue")
  val_loss_plot, = ax.plot(timerange, val_loss, color = "cyan")
  train_loss_plot.set_label("Train Loss")
  val_loss_plot.set_label("Validation Loss")
  ax.set_xlabel("Epoch")
  ax.set_ylabel("Loss")
  ax.legend(loc = "upper left")
  ax2 = ax.twinx()
  train_acc_plot, = ax2.plot(timerange, accuracy, color = "purple")
  val_acc_plot, = ax2.plot(timerange, val_accuracy, color = "pink")
  train_acc_plot.set_label("Train Accuracy")
  val_acc_plot.set_label("Validation Accuracy")
  ax2.set_ylabel("Accuracy")
  ax2.legend(loc = "upper right")
  plt.title("Loss vs Accuracy")
  plt.savefig(f"{dataset}_mgnet_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png")