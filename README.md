# two_tower_recommendation_system
A two tower recommendation system  implementation with tensorflow estimator.

基于tensorflow estimator API实现的双塔DNN推荐算法，可作为推荐算法模板在此基础上根据需求修改。

输入特征有：
    
    向量类特征： user向量、item向量
    
    分桶类特征： 年龄
    
    hash类特征： deviceID
    
 直接使用train_local.sh即可在本地训练，如果需要分布式训练，需设置train_on_cluster=True,然后提交到job中，由于每个公司的job提交命令不一样，这里就不贴出了。
