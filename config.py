import json, os, re, codecs
import tensorflow as tf
 
flags = tf.app.flags
 
flags.DEFINE_boolean("train_on_cluster", True, "Whether the cluster info need to be passed in as input")
 
flags.DEFINE_string("train_dir", "", "")
flags.DEFINE_string("data_dir", "", "")
flags.DEFINE_string("log_dir", "", "")
flags.DEFINE_string("ps_hosts", "","")
flags.DEFINE_string("worker_hosts", "","")
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
flags.DEFINE_string("model_dir", "hdfs:/user/ranker/ckpt/", "Base directory for the model.")
flags.DEFINE_string("output_model", "hdfs:/user/ranker/model/", "Saved model.")
flags.DEFINE_string("train_data", "./train_data.txt", "Directory for storing mnist data")
flags.DEFINE_string("eval_data", "./eval_data.txt", "Path to the evaluation data.")
flags.DEFINE_string("hidden_units", "512,256,128", "hidden units.")
flags.DEFINE_integer("train_steps",1000000, "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 256, "Training batch size")
flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
flags.DEFINE_integer("save_checkpoints_steps", 10000, "Save checkpoints every this many steps")
 
FLAGS = flags.FLAGS