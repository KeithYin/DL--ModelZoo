import tensorflow as tf
from tensorflow.contrib import layers
from collections import OrderedDict


class InceptionV1Blocks(object):
    def __init__(self):
        inceptions_paras = OrderedDict()
        inceptions_paras["inception3a"] = [64, 96, 128, 16, 32, 32]
        inceptions_paras["inception3b"] = [128, 128, 192, 32, 96, 64]
        inceptions_paras["inception4a"] = [192, 96, 208, 16, 48, 64]
        inceptions_paras["inception4b"] = [160, 112, 224, 24, 64, 64]
        inceptions_paras["inception4c"] = [128, 128, 256, 24, 64, 64]
        inceptions_paras["inception4d"] = [112, 144, 288, 32, 64, 64]
        inceptions_paras["inception4e"] = [256, 160, 320, 32, 128, 128]
        inceptions_paras["inception5a"] = [256, 160, 320, 32, 128, 128]
        inceptions_paras["inception5b"] = [384, 192, 384, 48, 128, 128]
        self.incep_v1_param = inceptions_paras

    def inference(self, inputs):
        """
        the inference part of the model
        :param inputs: Tensor [batch_size, height, width, channel]
        :return: logits
        """
        block_names = ["inception3a", "inception3b",
                       "inception4a", "inception4b", "inception4c",
                       "inception4d", "inception4e", "inception5a", "inception5b"]
        conv1 = layers.conv2d(inputs, num_outputs=64, kernel_size=7, stride=2, scope="conv1")
        pool1 = layers.max_pool2d(conv1, kernel_size=3, stride=2, padding="SAME",
                                  scope="pool1")
        norm1 = tf.nn.local_response_normalization(pool1, depth_radius=3, bias=2, alpha=10 ** (-4),
                                                   beta=0.75, name="LRM1")
        conv2_3_3_reduce = layers.conv2d(norm1, num_outputs=64, kernel_size=1, stride=1,
                                         scope="conv2_3_3_reduce")
        conv2_3_3 = layers.conv2d(conv2_3_3_reduce, num_outputs=192, kernel_size=3, stride=1,
                                  scope="conv2_3_3")
        norm2 = tf.nn.local_response_normalization(conv2_3_3, depth_radius=3, bias=2, alpha=10 ** (-4),
                                                   beta=0.75, name="LRM2")
        pool2 = layers.max_pool2d(norm2, kernel_size=3, stride=2, padding="SAME",
                                  scope="pool2")

        inception3a = InceptionV1Blocks.block(pool2,
                                              parameters=self.incep_v1_param[block_names[0]],
                                              variable_scope=block_names[0])
        inception3b = InceptionV1Blocks.block(inception3a,
                                              parameters=self.incep_v1_param[block_names[1]],
                                              variable_scope=block_names[1])
        pool3 = layers.max_pool2d(inception3b, kernel_size=3, stride=2, padding="SAME",
                                  scope="pool3")
        inception4a = InceptionV1Blocks.block(pool3,
                                              parameters=self.incep_v1_param[block_names[2]],
                                              variable_scope=block_names[2])
        inception4b = InceptionV1Blocks.block(inception4a,
                                              parameters=self.incep_v1_param[block_names[3]],
                                              variable_scope=block_names[3])
        inception4c = InceptionV1Blocks.block(inception4b,
                                              parameters=self.incep_v1_param[block_names[4]],
                                              variable_scope=block_names[4])
        inception4d = InceptionV1Blocks.block(inception4c,
                                              parameters=self.incep_v1_param[block_names[5]],
                                              variable_scope=block_names[5])
        inception4e = InceptionV1Blocks.block(inception4d,
                                              parameters=self.incep_v1_param[block_names[6]],
                                              variable_scope=block_names[6])
        pool4 = layers.max_pool2d(inception4e, kernel_size=3, stride=2, padding="SAME",
                                  scope="pool4")
        inception5a = InceptionV1Blocks.block(pool4,
                                              parameters=self.incep_v1_param[block_names[7]],
                                              variable_scope=block_names[7])
        inception5b = InceptionV1Blocks.block(inception5a,
                                              parameters=self.incep_v1_param[block_names[8]],
                                              variable_scope=block_names[8])
        avg_pool = layers.avg_pool2d(inception5b, kernel_size=7, stride=1, scope="avg_pool")

        return avg_pool  # shape [batch_size, 1, 1, 1024], you can reshape it a little bit

    @staticmethod
    def block(inputs, parameters, variable_scope):
        """
        using for generate block of inception v1
        :param inputs: Tensor the input of the block
        :param parameters: list of output size
        :param variable_scope: string , specify the variable scope name
        :return: Tensor
        """
        if not isinstance(parameters, list):
            raise ValueError("parameters must be a list")
        assert len(parameters) == 6

        with tf.variable_scope(variable_scope):
            out1_1 = layers.conv2d(inputs, num_outputs=parameters[0], kernel_size=1, stride=1)
            out3_3_reduce = layers.conv2d(inputs, num_outputs=parameters[1], kernel_size=1, stride=1)
            out3_3 = layers.conv2d(out3_3_reduce, num_outputs=parameters[2], kernel_size=3, stride=1)
            out5_5_reduce = layers.conv2d(inputs, num_outputs=parameters[3], kernel_size=1, stride=1)
            out5_5 = layers.conv2d(out5_5_reduce, num_outputs=parameters[4], kernel_size=5, stride=1)
            out_max_pool = layers.max_pool2d(inputs, kernel_size=3, stride=1, padding="SAME")
            out_proj = layers.conv2d(out_max_pool, num_outputs=parameters[5], kernel_size=1, stride=1)

            output = tf.concat([out1_1, out3_3, out5_5, out_proj], axis=-1)
        return output
