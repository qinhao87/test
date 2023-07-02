import tensorflow as tf
# 下面语句都可以达到检测目的
tf.test.is_gpu_available()
tf.config.list_physical_devices('GPU')
tf.test.gpu_device_name()