import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import numpy as np
import time
import os

def mlp_model_freeze(model_path, save_model_path):
    facenet_and_mlp_meta = '{}.meta'.format(model_path)
    facenet_and_mlp_ckpt = model_path
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(facenet_and_mlp_meta, clear_devices=True)
        saver.restore(sess, facenet_and_mlp_ckpt)
        
         
        #graph_def = tf.get_default_graph().as_graph_def()
        #output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['linear/add','Placeholder','input','embeddings','phase_train'])
            #output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['linear/add'])
        with tf.gfile.GFile('{}.pb'.format(os.path.join(save_model_path, model_path.split('/')[-1])), 'wb') as f:
            graph_def = tf.get_default_graph().as_graph_def()
            output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['linear/logits'])
            f.write(output_graph_def.SerializeToString())

checkpoint_path = '/data1/facenet-master/logs/fully_connected_feed/checkpoint'
save_model_path = '/data1/facenet-master/logs'

model_path = []
with open(checkpoint_path, 'rb') as f:
    for index, item in enumerate(f.readlines()):
        if index == 0:
            continue
        model_path.append(item.strip().split(' ')[-1][1:-1])
for index, item in enumerate(model_path):
    print(index, item)
    mlp_model_freeze(item, save_model_path=save_model_path)

# with gfile.FastGFile('/media/universe/768CE57C8CE53771/mnist/face_batch_demo/face_model/model.ckpt-9.pb', 'rb') as f:
#     graph_def_mlp = tf.GraphDef()
#     graph_def_mlp.ParseFromString(f.read())
#     mlp_logits, mlp_images_features_placehoder = tf.import_graph_def(graph_def_mlp,return_elements=['linear/logits:0','Placeholder:0'])
#
# with tf.Session() as sess:
#     for index in range(5000):
#         images_feature = np.ones((1,128))
#         t5 = time.time()
#         #images_result = sess.run(tf.nn.softmax(mlp_logits), {mlp_images_features_placehoder: images_feature})
#         images_result = sess.run(mlp_logits, {mlp_images_features_placehoder: images_feature})
#         t6 = time.time()
#         print('{} :{}'.format(index, t6 - t5))



