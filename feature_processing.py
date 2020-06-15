#coding:utf-8
import tensorflow as tf
from tensorflow import feature_column as fc
import config
 
FLAGS = config.FLAGS
 
class FeatureConfig(object):
    def __init__(self):
        self.user_columns = dict()
        self.item_columns = dict()
        
        self.feature_spec = dict()
 
    def create_features_columns(self):
        # 向量类特征
        user_vector = fc.numeric_column(key="user_vector", shape=(128,), default_value=[0.0] * 128, dtype=tf.float32)
        item_vector = fc.numeric_column(key="item_vector", shape=(128,), default_value=[0.0] * 128, dtype=tf.float32)
        
        # 分桶类特征
        age = fc.numeric_column(key="age", shape=(1,), default_value=[0], dtype=tf.int64)
        age = fc.bucketized_column(input_fc, boundaries=[0,10,20,30,40,50,60,70,80])
        age = fc.embedding_column(age, dimension=32, combiner='mean')
        
        # 分类特征
        city = fc.categorical_column_with_identity(key="city", num_buckets=1000, default_value=0)
        city = fc.embedding_column(city, dimension=32, combiner='mean')
        
        # hash特征
        device_id = fc.categorical_column_with_hash_bucket(key="device_id", hash_bucket_size=1000000, dtype=tf.int64)
        device_id = fc.embedding_column(device_id, dimension=32, combiner='mean')
        
        item_id = fc.categorical_column_with_hash_bucket(key="item_id", hash_bucket_size=10000, dtype=tf.int64)
        item_id = fc.embedding_column(device_id, dimension=32, combiner='mean')
        
        self.user_columns["user_vector"] = user_vector
        self.user_columns["age"] = age
        self.user_columns["city"] = city
        self.user_columns["device_id"] = device_id
        self.item_columns["item_vector"] = item_vector
        self.item_columns["item_id"] = item_id
 
        self.feature_spec = tf.feature_column.make_parse_example_spec(self.user_columns.values()+self.item_columns.values())
 
        return self