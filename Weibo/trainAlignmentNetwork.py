from idea02Function import FeatureMapper,CLIPWithFeatureMapping,train_model
from idea02Dataset import train_dataloader,test_dataloader





# 初始化线性层，假设我们将特征映射到512维
text_feature_mapper = FeatureMapper(768, 512)  # 768是BERT的默认输出维度
image_feature_mapper = FeatureMapper(1000, 512)  # 1000是ResNet的默认输出维度

# 实例化模型
clip_with_mapping = CLIPWithFeatureMapping( text_feature_mapper, image_feature_mapper)


metrics = train_model(train_dataloader,test_dataloader,model=clip_with_mapping)