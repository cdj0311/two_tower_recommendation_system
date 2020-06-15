# encoding:utf-8
import os
import tensorflow as tf
from tensorflow import feature_column as fc
import feature_engineering as fe
from feature_processing import FeatureConfig
import base_model
import data_input
import config
 
FLAGS = config.FLAGS
 
if FLAGS.run_on_cluster:
    cluster = json.loads(os.environ["TF_CONFIG"])
    task_index = int(os.environ["TF_INDEX"])
    task_type = os.environ["TF_ROLE"]
 
 
def main(unused_argv):
    feature_configs = FeatureConfig().create_features_columns()
    classifier = tf.estimator.Estimator(model_fn=model.model_fn,
                                        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                                                      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                                                      keep_checkpoint_max=3),
                                        params={"feature_configs": feature_configs,
                                                "hidden_units": list(map(int, FLAGS.hidden_units.split(","))),
                                                "learning_rate": FLAGS.learning_rate}
                                        )
    def train_eval_model():
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: data_input.train_input_fn(FLAGS.train_data, FLAGS.batch_size),
                                            max_steps=FLAGS.train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: data_input.eval_input_fn(FLAGS.eval_data, FLAGS.batch_size),
                                          start_delay_secs=60,
                                          throttle_secs = 30,
                                          steps=1000)
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
 
    def export_model():
        feature_spec = feature_configs.feature_spec
        feature_map = {}
        for key, feature in feature_spec.items():
            if key not in fe.feature_configs:
                continue
            if isinstance(feature, tf.io.VarLenFeature):  # 可变长度
                feature_map[key] = tf.placeholder(dtype=feature.dtype, shape=[1], name=key)
            elif isinstance(feature, tf.io.FixedLenFeature):  # 固定长度
                feature_map[key] = tf.placeholder(dtype=feature.dtype, shape=[None, feature.shape[0]], name=key)
        serving_input_recevier_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_map)
        export_dir = classifier.export_saved_model(FLAGS.output_model, serving_input_recevier_fn)
 
    # 模型训练
    train_eval_model()
    
    # 导出模型，只在chief中导出一次即可
    if FLAGS.train_on_cluster: 
        if task_type == "chief":
            export_model()
    else:
        export_model()
 
 
if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)