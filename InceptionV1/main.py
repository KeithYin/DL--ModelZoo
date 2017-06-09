import tensorflow as tf
from models.blocks import InceptionV1Blocks



def main(_):
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
    inceptionV1Blocks = InceptionV1Blocks()

    feature = inceptionV1Blocks.inference(inputs)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./checkpoint", sess.graph)

if __name__ == '__main__':
    tf.app.run()