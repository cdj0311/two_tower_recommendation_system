# coding:utf-8
import tensorflow as tf
from tensorflow import feature_column as fc
import config
import feature_processing as fe
 
FLAGS = config.FLAGS
 
 
def build_user_model(features, mode, params):
    # 特征输入
    feature_inputs = []
    for key, value in params["feature_configs"].user_columns.items():
        feature_inputs.append(tf.input_layer(features, value))
    # 特征拼接
    net = tf.concat(feature_inputs, axis=1)
    # 全连接
    for idx, units in enumerate(params["hidden_units"]):
        net = tf.layers.dense(net, units=units, activation=tf.nn.leaky_relu, name="user_fc_layer_%s"%idx)
        net = tf.layers.dropout(net, 0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
    # 最后输出
    net = tf.layers.dense(net, units=128, name="user_output_layer")
    return net
    
def build_item_model(features, mode, params):
    # 特征输入
    feature_inputs = []
    for key, value in params["feature_configs"].item_columns.items():
        feature_inputs.append(tf.input_layer(features, value))
    # 特征拼接
    net = tf.concat(feature_inputs, axis=1)
    # 全连接
    for idx, units in enumerate(params["hidden_units"]):
        net = tf.layers.dense(net, units=units, activation=tf.nn.leaky_relu, name="user_fc_layer_%s"%idx)
        net = tf.layers.dropout(net, 0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
    # 最后输出
    net = tf.layers.dense(net, units=128, name="user_output_layer")
    return net
 
def model_fn(features, labels, mode, params):
    user_net = build_user_model(features, mode, params)
    item_net = build_item_model(features, mode, params)
    dot = tf.reduce_sum(tf.multiply(user_net, item_net), axis=1, keepdims=True)
    pred = tf.sigmoid(dot)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={"output": pred})
    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.log_loss(labels, pred)
        metrics = {"auc": tf.metrics.auc(labels=labels, predictions=pred)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.log_loss(labels, pred)
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(params["learning_rate"], global_step, 100000, 0.9, staircase=True)
        train_op = (tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step))
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
