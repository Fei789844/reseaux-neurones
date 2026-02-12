library(kohonen)
library(cluster) 
library(ggplot2)
penguins <- read.csv("~/Downloads//神经网络Projet/penguins.csv")
penguin_data <- penguins[, c("culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g")]
penguin_data_clean <- na.omit(penguin_data)
penguin_data_scaled <- scale(penguin_data_clean)

som_grid <- somgrid(xdim = 2, ydim = 2, topo = "hexagonal")
som_model <- som(penguin_data_scaled, grid = som_grid)

# 获取SOM的聚类结果
som_clusters <- som_model$unit.classif  # 长度应该是342，与数据点数量一致

# 将聚类结果与原始数据对应
som_df <- data.frame(penguin_data_scaled)  # 使用原始数据框进行匹配
som_df$cluster <- as.factor(som_clusters)  # 将聚类结果添加到数据框

# 使用更多主成分进行PCA可视化
pca <- prcomp(som_df[, c("culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g")])
pca_df <- data.frame(PC1 = pca$x[, 1], PC2 = pca$x[, 2] , PC3 = pca$x[, 3], PC4 = pca$x[, 4], Cluster = som_df$cluster)

ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 4) +
  labs(title = "SOM Clustering - PCA Visualization", x = "PC1", y = "PC2") +
  theme_minimal()
plot(som_model, type = "codes")  # 绘制网格点图
plot(som_model, type = "mapping", col = som_df$cluster, main = "SOM Clustering - Grid Visualization")

ggplot(pca_df, aes(x = PC3, y = PC4, color = Cluster)) +
  geom_point(size = 4) +
  labs(title = "SOM Clustering - PCA Visualization (PC3 vs PC4)", x = "PC3", y = "PC4") +
  theme_minimal()
library(ggplot2) 
library(cluster)
sil <- silhouette(som_clusters, dist(penguin_data_scaled))  # som_clusters 是您的SOM聚类结果
ggplot(sil)  # 绘制轮廓系数图，越接近1，聚类效果越好



# 1. 确保标准化数据
penguin_data_scaled <- scale(penguin_data_clean)

# 2. 检查SOM聚类数量，减少聚类数量尝试
som_grid <- somgrid(xdim = 5, ydim = 5, topo = "hexagonal")  # 减少网格点
som_model <- som(penguin_data_scaled, grid = som_grid)

# 3. 绘制SOM网格
plot(som_model, type = "mapping", col = som_df$cluster, main = "SOM Clustering - Grid Visualization")

# 4. PCA降维并可视化，增加PC3和PC4
pca <- prcomp(som_df[, c("culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g")])
pca_df <- data.frame(PC1 = pca$x[, 1], PC2 = pca$x[, 2], PC3 = pca$x[, 3], PC4 = pca$x[, 4], Cluster = som_df$cluster)

# 可视化PC1和PC2
ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 4) +
  labs(title = "SOM Clustering - PCA Visualization", x = "PC1", y = "PC2") +
  theme_minimal()

# 也可以尝试PC3和PC4
ggplot(pca_df, aes(x = PC3, y = PC4, color = Cluster)) +
  geom_point(size = 4) +
  labs(title = "SOM Clustering - PCA Visualization (PC3 vs PC4)", x = "PC3", y = "PC4") +
  theme_minimal()
library(ggplot2) 
library(cluster)
sil <- silhouette(som_clusters, dist(penguin_data_scaled))  # som_clusters 是您的SOM聚类结果
ggplot(sil)  # 绘制轮廓系数图，越接近1，聚类效果越好



# 1. 加载必需包
library(kohonen)  # SOM核心包
library(cluster)   # 辅助聚类（k-means二次聚类）
library(ggplot2)   # 可视化
library(dplyr)     # 数据处理（可选，简化操作）

# 2. 数据读取与预处理
# 注意：请根据你的实际文件路径修改（删除多余的//，避免路径错误）
penguins <- read.csv("~/Downloads/神经网络Projet/penguins.csv")

# 提取数值特征（4个形态学特征）
feature_cols <- c("culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g")
penguin_data <- penguins[, feature_cols]

# 数据清洗：删除缺失值
penguin_data_clean <- na.omit(penguin_data)

# 特征缩放（关键！SOM对特征尺度极其敏感，必须标准化）
penguin_data_scaled <- scale(penguin_data_clean)  # 均值为0，方差为1

# 3. 构建SOM模型（核心步骤，合理设置网格尺寸）
# 网格尺寸：不使用10x10（类别过多），选择5x5（25个神经元，后续二次聚类为3-5类更合理）
# 拓扑结构：hexagonal（六边形，比矩形更均匀的邻居分布，SOM常用）
som_grid <- somgrid(xdim = 5, ydim = 5, topo = "hexagonal", toroidal = FALSE)

# 训练SOM模型
som_model <- som(
  data = penguin_data_scaled,
  grid = som_grid,
  rlen = 1000,  # 训练迭代次数（足够迭代保证收敛）
  alpha = c(0.05, 0.01),  # 学习率从0.05衰减到0.01，保证稳定收敛
  keep.data = TRUE  # 保留原始数据，方便后续可视化
)

# 4. SOM结果优化：对神经元进行二次聚类（解决原始SOM类别过多问题）
# 提取SOM每个神经元的特征向量（码本向量）
som_codebook <- som_model$codes[[1]]  # 每行对应一个神经元的特征

# 使用k-means对神经元进行聚类（聚成3类，对应企鹅的3个物种，更具解释性）
k <- 3  # 聚类数量（可根据需求调整为2-5）
kmeans_result <- kmeans(som_codebook, centers = k, nstart = 20)  # nstart避免局部最优

# 将每个样本映射到对应的二次聚类类别（核心：样本→获胜神经元→k-means类别）
# som_model$unit.classif：每个样本对应的获胜神经元编号
# kmeans_result$cluster：每个神经元对应的k-means聚类类别
penguin_som_clusters <- kmeans_result$cluster[som_model$unit.classif]

# 构建包含聚类结果的数据框
som_result_df <- cbind(
  penguin_data_clean,  # 原始清洗后的数据
  som_neuron = as.factor(som_model$unit.classif),  # 样本对应的获胜神经元
  som_cluster = as.factor(penguin_som_clusters)     # 样本对应的最终SOM-二次聚类类别
)

# 5. 可视化1：PCA可视化（展示样本聚类分布，解决之前黏连问题）
# PCA时开启特征缩放（即使原始数据已缩放，此处再次确认，确保PC轴无尺度偏差）
pca_model <- prcomp(penguin_data_clean, scale. = TRUE)

# 提取前两个主成分（解释最大方差，可视化效果最佳）
pca_df <- data.frame(
  PC1 = pca_model$x[, 1],
  PC2 = pca_model$x[, 2],
  SOM_Cluster = som_result_df$som_cluster,
  SOM_Neuron = som_result_df$som_neuron
)

# 绘制PCA聚类散点图
pca_plot <- ggplot(pca_df, aes(x = PC1, y = PC2, color = SOM_Cluster)) +
  geom_point(size = 3, alpha = 0.8) +  # alpha增加透明度，避免点重叠遮挡
  labs(
    title = "SOM聚类结果 - PCA可视化",
    x = paste0("PC1 (", round(summary(pca_model)$importance[2,1]*100, 1), "%)"),
    y = paste0("PC2 (", round(summary(pca_model)$importance[2,2]*100, 1), "%)"),
    color = "SOM聚类类别"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
print(pca_plot)

# 6. 可视化2：SOM专属核心可视化（3种关键图，体现SOM特性）
# 6.1 SOM神经元特征图（type = "codes"：展示神经元码本向量分布）
par(mfrow = c(1, 2))  # 布局：1行2列
# 神经元特征图（按k-means聚类着色，展示神经元分组）
plot(som_model,
     type = "codes",
     bgcol = terrain.colors(k)[kmeans_result$cluster],  # 聚类专属颜色
     main = "SOM神经元特征分布（按聚类着色）",
     palette.name = terrain.colors)

# 6.2 SOM样本映射图（type = "mapping"：展示每个神经元上的样本分布）
plot(som_model,
     type = "mapping",
     col = as.integer(som_result_df$som_cluster),  # 样本按聚类着色
     pch = 19,  # 点样式
     main = "SOM样本映射图（每个点代表一个企鹅样本）",
     palette.name = terrain.colors)
par(mfrow = c(1, 1))  # 恢复默认布局

# 6.3 SOM神经元样本计数图（type = "counts"：展示每个神经元承载的样本数量）
plot(som_model,
     type = "counts",
     main = "SOM神经元样本承载量",
     palette.name = heat.colors,  # 热图颜色，数值越高颜色越深
     main.font = 2)

# 7. 输出关键结果
cat("SOM模型训练完成！\n")
cat("样本总数：", nrow(som_result_df), "\n")
cat("SOM神经元数量：", nrow(som_codebook), "\n")
cat("最终聚类类别数：", k, "\n")
# 输出每个聚类的样本数量
table(som_result_df$som_cluster)

