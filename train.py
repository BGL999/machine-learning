import os
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
import warnings

warnings.filterwarnings('ignore')


class MultiFeaturePlantClassifier:
    def __init__(self, data_path, img_size=(128, 128)):
        """
        初始化多特征植物分类器
        Args:
            data_path: 数据集路径
            img_size: 统一调整的图像尺寸
        """
        self.data_path = data_path
        self.img_size = img_size
        self.classes = []
        self.features = []
        self.labels = []
        self.scaler = StandardScaler()
        self.pca = None
        self.model = None
        self.best_params_ = None

    def load_all_images(self):
        """加载所有图像数据"""
        print("正在加载所有图像数据...")
        self.classes = sorted([d for d in os.listdir(self.data_path)
                               if os.path.isdir(os.path.join(self.data_path, d))])

        if not self.classes:
            raise ValueError(f"在路径 {self.data_path} 中没有找到类别文件夹")

        print(f"找到 {len(self.classes)} 个类别: {self.classes}")

        total_images = 0
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_path, class_name)

            # 获取所有图像文件
            import glob
            image_files = glob.glob(os.path.join(class_path, '*.png')) + \
                          glob.glob(os.path.join(class_path, '*.jpg')) + \
                          glob.glob(os.path.join(class_path, '*.jpeg'))

            print(f"\n处理类别 '{class_name}' ({len(image_files)} 张图片):")

            for i, img_path in enumerate(image_files):
                if total_images % 50 == 0 and total_images > 0:
                    print(f"  已处理 {total_images} 张图片...")

                # 提取多特征
                features = self.extract_multi_features(img_path)
                if features is not None:
                    self.features.append(features)
                    self.labels.append(class_idx)
                    total_images += 1

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)

        print(f"\n{'=' * 60}")
        print(f"数据加载完成:")
        print(f"  总图片数: {len(self.features)}")
        print(f"  特征维度: {self.features.shape}")

        # 显示类别分布
        print(f"  类别分布:")
        for i, class_name in enumerate(self.classes):
            count = np.sum(self.labels == i)
            print(f"    {class_name}: {count} 张图片")
        print('=' * 60)

    def extract_multi_features(self, img_path):
        """
        提取多种特征融合
        Args:
            img_path: 图像路径
        Returns:
            融合特征向量
        """
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图像: {img_path}")
                return None

            # 调整图像大小
            img_resized = cv2.resize(img, self.img_size)
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            feature_vector = []

            # 1. HOG特征（形状和边缘特征）
            hog_features = self.extract_hog_features(gray)
            feature_vector.extend(hog_features)

            # 2. 颜色直方图特征（颜色分布）
            color_features = self.extract_color_features(img_resized)
            feature_vector.extend(color_features)

            # 3. LBP特征（局部纹理特征）
            lbp_features = self.extract_lbp_features(gray)
            feature_vector.extend(lbp_features)

            # 4. Hu矩（形状不变特征）
            hu_features = self.extract_hu_moments(gray)
            feature_vector.extend(hu_features)

            # 5. 边缘特征统计
            edge_features = self.extract_edge_features(gray)
            feature_vector.extend(edge_features)

            return np.array(feature_vector)

        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            return None

    def extract_hog_features(self, gray_img):
        """提取HOG特征"""
        # 使用较大的单元格减少特征维度
        features = hog(
            gray_img,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            feature_vector=True
        )
        return features

    def extract_color_features(self, color_img):
        """提取颜色直方图特征"""
        features = []

        # RGB颜色直方图
        for i in range(3):
            hist = cv2.calcHist([color_img], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

        # HSV颜色空间（对光照变化更鲁棒）
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        for i in range(3):
            if i == 0:  # Hue通道
                hist = cv2.calcHist([hsv], [i], None, [32], [0, 180])
            else:
                hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

        return features

    def extract_lbp_features(self, gray_img):
        """提取LBP纹理特征"""
        # 计算LBP
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')

        # 计算直方图
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # 归一化

        return hist

    def extract_hu_moments(self, gray_img):
        """提取Hu矩特征"""
        moments = cv2.moments(gray_img)
        hu_moments = cv2.HuMoments(moments).flatten()
        # 对数变换增强数值稳定性
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        return hu_moments

    def extract_edge_features(self, gray_img):
        """提取边缘特征统计"""
        # Sobel边缘检测
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

        sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        edge_features = [
            np.mean(sobel_magnitude),
            np.std(sobel_magnitude),
            np.max(sobel_magnitude),
            np.sum(sobel_magnitude > np.mean(sobel_magnitude)) / sobel_magnitude.size
        ]

        return edge_features

    def preprocess_features(self):
        """特征预处理"""
        print("\n正在进行特征预处理...")

        # 标准化特征
        X_scaled = self.scaler.fit_transform(self.features)

        # PCA降维
        self.pca = PCA(n_components=0.95, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)

        print(f"原始特征维度: {X_scaled.shape[1]}")
        print(f"PCA降维后维度: {X_pca.shape[1]}")
        print(f"累计解释方差: {self.pca.explained_variance_ratio_.sum():.3f}")

        return X_pca

    def optimize_and_train_final_model(self):
        """
        使用网格搜索优化参数并训练最终模型
        """
        print("\n" + "=" * 60)
        print("使用网格搜索优化SVM参数并训练最终模型")
        print("=" * 60)

        # 预处理所有特征
        X = self.preprocess_features()
        y = self.labels

        print(f"数据准备完成:")
        print(f"  总样本数: {X.shape[0]}")
        print(f"  特征维度: {X.shape[1]}")

        # 基于上次的最佳结果调整参数网格
        param_grid = {
            'C': [0.1, 1, 10, 100],  # 扩展C的范围
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  # RBF核的gamma参数
            'kernel': ['rbf']  # 只使用RBF核，上次效果最好
        }

        # 创建SVM分类器
        svm = SVC(random_state=42, class_weight='balanced')

        print("\n开始网格搜索优化参数...")
        print(f"参数组合总数: {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])}")
        print("这可能需要一些时间，请耐心等待...")

        # 使用交叉验证进行网格搜索
        grid_search = GridSearchCV(
            svm,
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=1,  # 设置为1，避免并行处理导致内存问题
            verbose=2,  # 显示详细进度
            return_train_score=True
        )

        # 执行网格搜索
        grid_search.fit(X, y)

        # 保存最佳参数
        self.best_params_ = grid_search.best_params_

        print(f"\n网格搜索完成!")
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

        # 使用最佳模型作为最终模型
        self.model = grid_search.best_estimator_

        # 显示交叉验证性能
        self.show_cv_performance(grid_search)

        # 分析特征贡献
        self.analyze_feature_contributions()

        return self.model

    def show_cv_performance(self, grid_search):
        """显示交叉验证性能"""
        print(f"\n{'=' * 60}")
        print("交叉验证性能分析")
        print('=' * 60)

        cv_results = grid_search.cv_results_

        # 显示前5个最佳参数组合
        top_indices = np.argsort(cv_results['mean_test_score'])[-5:][::-1]

        print("\n前5个最佳参数组合:")
        for i, idx in enumerate(top_indices):
            print(f"{i + 1}. 参数: {cv_results['params'][idx]}")
            print(f"   平均验证分数: {cv_results['mean_test_score'][idx]:.4f}")
            print(f"   标准差: {cv_results['std_test_score'][idx]:.4f}")

        # 显示最佳模型的交叉验证分数分布
        best_idx = grid_search.best_index_
        best_scores = []
        for i in range(grid_search.cv.n_splits):
            score_key = f'split{i}_test_score'
            if score_key in cv_results:
                best_scores.append(cv_results[score_key][best_idx])

        print(f"\n最佳模型的5折交叉验证分数:")
        for i, score in enumerate(best_scores):
            print(f"  第{i + 1}折: {score:.4f}")
        print(f"  平均值: {np.mean(best_scores):.4f}")
        print(f"  标准差: {np.std(best_scores):.4f}")

    def analyze_feature_contributions(self):
        """分析不同特征类型的贡献"""
        print(f"\n{'=' * 60}")
        print("特征类型贡献分析")
        print('=' * 60)

        # 特征维度信息（根据提取函数的实现）
        feature_dimensions = {
            'HOG特征': 1764,  # 128x128图像，pixels_per_cell=(16,16)时的HOG特征维度
            '颜色直方图': 192,  # 32bin * 6通道 (RGB+HSV)
            'LBP纹理': 59,  # uniform LBP特征
            'Hu矩': 7,  # 7个Hu矩
            '边缘统计': 4  # 4个边缘统计特征
        }

        total_dim = sum(feature_dimensions.values())

        print("\n各特征类型维度分布:")
        for feat_name, dim in feature_dimensions.items():
            percentage = dim / total_dim * 100
            print(f"  {feat_name}: {dim} 维 ({percentage:.1f}%)")

        print(f"  总计: {total_dim} 维")

        # 可视化特征维度分布
        labels = list(feature_dimensions.keys())
        sizes = list(feature_dimensions.values())

        plt.figure(figsize=(10, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('特征类型维度分布', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def save_model(self, filename='final_multi_feature_classifier.pkl'):
        """保存模型"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'pca': self.pca,
                'classes': self.classes,
                'img_size': self.img_size,
                'best_params': self.best_params_,
                'feature_types': ['HOG', '颜色直方图', 'LBP', 'Hu矩', '边缘统计']
            }, filename)
            print(f"\n模型已保存为 '{filename}'")
        else:
            print("警告: 没有训练好的模型可以保存")


def main():
    """
    主函数：执行多特征植物分类任务
    """
    # 设置数据集路径
    data_path = '/kaggle/input/poject1/dataset-for-task1/dataset-for-task1/train'

    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"错误：路径 {data_path} 不存在！")
        print("请修改为正确的数据集路径")
        return

    print("=" * 70)
    print("植物分类系统 - 多特征融合 + SVM模型")
    print("特征包括: HOG, 颜色直方图, LBP, Hu矩, 边缘统计")
    print("使用网格搜索优化参数")
    print("=" * 70)

    # 创建分类器
    classifier = MultiFeaturePlantClassifier(data_path, img_size=(128, 128))

    try:
        # 1. 加载所有数据
        classifier.load_all_images()

        # 2. 使用网格搜索优化参数并训练最终模型
        final_model = classifier.optimize_and_train_final_model()

        # 3. 保存模型
        classifier.save_model()

        print("\n" + "=" * 70)
        print("植物分类任务完成！")
        print(f"特征工程方法: 多特征融合")
        print(f"  1. HOG特征 (形状和边缘)")
        print(f"  2. 颜色直方图 (颜色分布)")
        print(f"  3. LBP特征 (局部纹理)")
        print(f"  4. Hu矩 (形状不变特征)")
        print(f"  5. 边缘统计 (边缘特征)")
        print(f"分类模型: SVM（支持向量机）")
        print(f"参数优化: 网格搜索")
        print(f"每个类别的100张图片已全部用于学习")
        print(f"模型已保存为 'final_multi_feature_classifier.pkl'")
        print("=" * 70)

    except MemoryError:
        print("\n内存不足！尝试以下解决方案:")
        print("1. 减小图像尺寸: 将 img_size=(128, 128) 改为 (64, 64)")
        print("2. 减少特征维度: 修改特征提取参数")
        print("3. 关闭其他程序释放内存")
    except Exception as e:
        print(f"\n运行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()