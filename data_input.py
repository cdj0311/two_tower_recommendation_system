# coding:utf-8
import tensorflow as tf
import config
 
FLAGS = config.FLAGS
 
def parse_exp(example):
    features_def = {}
    features_def["label"] = tf.io.FixedLenFeature([1], tf.int64)
    """
    数据解析逻辑
    """
    features = tf.io.parse_single_example(example, features_def)
    label = features["label"]
    return features, label
 
 
def train_input_fn(filenames=None, batch_size=128):
    with tf.gfile.Open(filenames) as f:
        filenames = f.read().split()
   
    if FLAGS.train_on_cluster:
        files_all = []
        for f in filenames:
            files_all += tf.gfile.Glob(f)
        train_worker_num = len(FLAGS.worker_hosts.split(","))
        hash_id = FLAGS.task_index if FLAGS.job_name == "worker" else train_worker_num - 1
        files_shard = [files for i, files in enumerate(files_all) if i % train_worker_num == hash_id]
        dataset = tf.data.TFRecordDataset(files_shard)
    else:
        files = tf.data.Dataset.list_files(filenames)
        dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.shuffle(batch_size*10)
    dataset = dataset.map(parse_exp, num_parallel_calls=8)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset
 
def eval_input_fn(filenames=None, batch_size=128):
    with tf.gfile.Open(filenames) as f:
        filenames = f.read().split()
    files = tf.data.Dataset.list_files(filenames)
    dataset = files.apply(tf.contrib.data.parallel_interleave(lambda filename: tf.data.TFRecordDataset(files), buffer_output_elements=batch_size*20, cycle_length=10))
    dataset = dataset.map(parse_exp, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    return dataset