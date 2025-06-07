import torch
import torch.optim as optim
from mymodel import test_dataloader,train_dataloader,FeatureMapper,CLIPWithFeatureMapping,CustomClassifiernoExternal
from transformers import  CLIPModel
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from common_function import compute_similarity_matrix
import numpy as np



def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average=None)
    return precision, recall, f1



# 加载模型和预处理器
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)

# 初始化线性层，假设我们将特征映射到512维
text_feature_mapper = FeatureMapper(768, 512)  # 768是BERT的默认输出维度
image_feature_mapper = FeatureMapper(1000, 512)  # 1000是ResNet的默认输出维度

# 实例化模型
clip_with_mapping = CLIPWithFeatureMapping(clip_model, text_feature_mapper, image_feature_mapper)
# 加载模型
model_path = "/home/zhouqing/codes/COOLANT/twitter/model_file/part01_bestclip_model_f1_0.9860.pt"
checkpoint = torch.load(model_path)

clip_with_mapping.load_state_dict(checkpoint['model_state_dict'])
# text_feature_mapper.load_state_dict(checkpoint['optimizer_state_dict'])







        
# 模型和训练参数
img_feature_dim = 512
ext_feature_dim = 512
txt_feature_dim = 512
fusion_dim = 1024
num_classes = 2
num_epochs = 70
learning_rate = 0.001

# 创建模型、损失函数和优化器
model = CustomClassifiernoExternal(img_feature_dim, txt_feature_dim, fusion_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
clip_with_mapping = clip_with_mapping.to(device)
clip_with_mapping.eval()  # 设置为评估模式

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_samples = 0
    all_labels = []
    all_preds = []

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    for tweetIds, tweetTexts, tweetText_features, imageIds, img_features, external_features, labels in progress_bar:
        # 将数据移动到 GPU（如果有的话）
        img_features = img_features.to(device)
        external_features = external_features.to(device)
        tweetText_features = tweetText_features.to(device)
        labels = labels.to(device)
        
        fused_features = external_features * tweetText_features
        mapped_text_features, mapped_image_features = clip_with_mapping(tweetText_features, img_features)
        mapped_fused_features, mapped_image_features02 = clip_with_mapping(fused_features, img_features)
        mapped_ex_features, mapped_image_features03 = clip_with_mapping(external_features, img_features)
        
        mapped_text_features_orin = mapped_text_features
        mapped_image_features_orin = mapped_image_features
        mapped_ex_features_ori = mapped_ex_features
        
        batch_size = mapped_text_features.size(0)
        mapped_text_features = mapped_text_features.detach().cpu().numpy().reshape(batch_size, -1)
        mapped_image_features = mapped_image_features.detach().cpu().numpy()
        mapped_fused_features = mapped_fused_features.detach().cpu().numpy().reshape(batch_size, -1)
        mapped_image_features02 = mapped_image_features02.detach().cpu().numpy()
        mapped_ex_features = mapped_ex_features.detach().cpu().numpy().reshape(batch_size, -1)
        mapped_image_features03 = mapped_image_features03.detach().cpu().numpy()
        
        similarity_matrix = compute_similarity_matrix(mapped_text_features, mapped_image_features)
        fused_similarity_matrix = compute_similarity_matrix(mapped_ex_features, mapped_image_features03)
        manual_predictions = []
        for i in range(batch_size):
            similarity_score = similarity_matrix[i, i]
            fused_similarity_score = fused_similarity_matrix[i, i]
            
            if similarity_score > 0.95 :
                # Both scores are greater than 0.5, predict real news (label 0)
                manual_predictions.append(0)
            elif similarity_score <= -1.5 :
                # Both scores are less than or equal to 0.5, predict fake news (label 1)
                manual_predictions.append(1)
            else:
                # Further model-based prediction is required
                outputs = model(mapped_image_features_orin[i:i+1], mapped_text_features_orin[i:i+1])
                pred_label = torch.argmax(outputs, dim=1).item()  # Get the predicted label (0 or 1)
                manual_predictions.append(pred_label)

        
        manual_predictions = torch.tensor(manual_predictions, dtype=torch.long, device=device)

        # Using model outputs for loss computation
        outputs = model(mapped_image_features_orin, mapped_text_features_orin)
        loss = criterion(outputs, labels)  # 'outputs' are logits, 'labels' are target class indices

        # 清零梯度
        optimizer.zero_grad()

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(manual_predictions.cpu().numpy())

        # 更新进度条的描述
        progress_bar.set_postfix({"loss": running_loss / total_samples})


    # 计算指标
    precision, recall, f1 = calculate_metrics(all_labels, all_preds)
    precision_real, precision_fake = precision
    recall_real, recall_fake = recall
    f1_real, f1_fake = f1
    print(f'Train Precision (Real): {precision_real:.4f}, Recall (Real): {recall_real:.4f}, F1 Score (Real): {f1_real:.4f}')
    print(f'Train Precision (Fake): {precision_fake:.4f}, Recall (Fake): {recall_fake:.4f}, F1 Score (Fake): {f1_fake:.4f}')

    print(f'Epoch [{epoch+1}/{num_epochs}]')
    # pall_predictions
    if (epoch + 1) % 5 == 0:
        model.eval()
        test_labels = []
        test_preds = []

        with torch.no_grad():
            for tweetIds, tweetTexts, tweetText_features, imageIds, img_features, external_features, labels in test_dataloader:
                img_features = img_features.to(device)
                external_features = external_features.to(device)
                tweetText_features = tweetText_features.to(device)
                labels = labels.to(device)
                
                fused_features = external_features * tweetText_features
                mapped_text_features, mapped_image_features = clip_with_mapping(tweetText_features, img_features)
                mapped_fused_features, mapped_image_features02 = clip_with_mapping(fused_features, img_features)
                mapped_ex_features, mapped_image_features03 = clip_with_mapping(external_features, img_features)
                
                mapped_text_features_orin = mapped_text_features
                mapped_image_features_orin = mapped_image_features
                mapped_ex_features_ori = mapped_ex_features
                
                batch_size = mapped_text_features.size(0)
                mapped_text_features = mapped_text_features.detach().cpu().numpy().reshape(batch_size, -1)
                mapped_image_features = mapped_image_features.detach().cpu().numpy()
                mapped_fused_features = mapped_fused_features.detach().cpu().numpy().reshape(batch_size, -1)
                mapped_image_features02 = mapped_image_features02.detach().cpu().numpy()
                mapped_ex_features = mapped_ex_features.detach().cpu().numpy().reshape(batch_size, -1)
                mapped_image_features03 = mapped_image_features03.detach().cpu().numpy()
                
                similarity_matrix = compute_similarity_matrix(mapped_text_features, mapped_image_features)
                fused_similarity_matrix = compute_similarity_matrix(mapped_ex_features, mapped_image_features03)
                manual_predictions = []
                for i in range(batch_size):
                    similarity_score = similarity_matrix[i, i]
                    fused_similarity_score = fused_similarity_matrix[i, i]
                    
                    if similarity_score > 0.95 :
                        # Both scores are greater than 0.5, predict real news (label 0)
                        manual_predictions.append(0)
                    elif similarity_score <= -0.6 :
                        # Both scores are less than or equal to 0.5, predict fake news (label 1)
                        manual_predictions.append(1)
                    else:
                        # Further model-based prediction is required
                        outputs = model(mapped_image_features_orin[i:i+1], mapped_text_features_orin[i:i+1])
                        pred_label = torch.argmax(outputs, dim=1).item()  # Get the predicted label (0 or 1)
                        manual_predictions.append(pred_label)
                
                manual_predictions = torch.tensor(manual_predictions, dtype=torch.long, device=device)
                outputs = model(mapped_image_features_orin, mapped_text_features_orin)
                loss = criterion(outputs, labels)  # 'outputs' are logits, 'labels' are target class indices

                running_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
                test_labels.extend(labels.cpu().numpy())
                test_preds.extend(manual_predictions.cpu().numpy())
                
                
                

                
        # 计算测试指标
        precision, recall, f1 = calculate_metrics(test_labels, test_preds)
        precision_real, precision_fake = precision
        recall_real, recall_fake = recall
        f1_real, f1_fake = f1
        # 计算准确率
        correct_predictions = (np.array(all_labels) == np.array(all_preds)).sum()
        accuracy = correct_predictions / len(all_labels)
        print(f'Test Accuracy: {accuracy:.4f}')

        print(f'Test Precision (Real): {precision_real:.4f}, Recall (Real): {recall_real:.4f}, F1 Score (Real): {f1_real:.4f}')
        print(f'Test Precision (Fake): {precision_fake:.4f}, Recall (Fake): {recall_fake:.4f}, F1 Score (Fake): {f1_fake:.4f}')


print('Finished Training')
        

        

