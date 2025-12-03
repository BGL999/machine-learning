import os
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
import joblib
import warnings
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')


class OptimizedPlantClassifier:
    def __init__(self, data_path, img_size=(64, 64)):  # 减小图片尺寸以减少计算量
        """
        初始化优化植物分类器
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
        self.label_encoder = LabelEncoder()
        self.feature_names = []  # 用于记录特征名称

    def load_all_images(self, use_augmentation=False):
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

            # 使用tqdm显示进度
            for img_path in tqdm(image_files, desc=f"处理{class_name}"):
                # 提取核心特征（减少特征维度）
                features = self.extract_optimized_features(img_path)
                if features is not None:
                    self.features.append(features)
                    self.labels.append(class_idx)
                    total_images += 1

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)

        # 编码标签
        self.labels = self.label_encoder.fit_transform(self.labels)

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

    def extract_optimized_features(self, img_path):
        """
        提取优化的特征集 - 减少特征维度，聚焦关键特征
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

            # 1. 颜色特征（选择最重要的一种颜色空间）
            color_features = self.extract_color_features(img_resized)
            feature_vector.extend(color_features)

            # 2. HOG特征（只使用一种尺度）
            hog_features = self.extract_hog_features(gray)
            feature_vector.extend(hog_features)

            # 3. LBP特征（单一尺度）
            lbp_features = self.extract_lbp_features(gray)
            feature_vector.extend(lbp_features)

            # 4. 形状特征（Hu矩）
            shape_features = self.extract_shape_features(gray)
            feature_vector.extend(shape_features)

            # 5. 边缘特征
            edge_features = self.extract_edge_features(gray)
            feature_vector.extend(edge_features)

            return np.array(feature_vector)

        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            return None

    def extract_color_features(self, color_img):
        """提取优化的颜色特征"""
        features = []

        # RGB颜色直方图（减少bins数量）
        for i in range(3):
            hist = cv2.calcHist([color_img], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

        # HSV颜色空间的统计特征
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        hsv_features = []
        for i in range(3):
            channel = hsv[:, :, i]
            hsv_features.extend([np.mean(channel), np.std(channel), np.median(channel)])
        features.extend(hsv_features)

        return features

    def extract_hog_features(self, gray_img):
        """提取HOG特征"""
        # 使用中等单元格大小
        hog_features = hog(
            gray_img,
            orientations=8,  # 减少方向数
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            feature_vector=True
        )
        return hog_features

    def extract_lbp_features(self, gray_img):
        """提取LBP特征"""
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist

    def extract_shape_features(self, gray_img):
        """提取形状特征"""
        features = []

        # Hu矩
        moments = cv2.moments(gray_img)
        hu_moments = cv2.HuMoments(moments).flatten()

        # 对Hu矩进行对数变换
        for moment in hu_moments:
            if moment != 0:
                features.append(-np.sign(moment) * np.log10(abs(moment)))
            else:
                features.append(0)

        return features

    def extract_edge_features(self, gray_img):
        """提取边缘特征"""
        features = []

        # Sobel边缘
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        edge_features = [
            np.mean(sobel_magnitude),
            np.std(sobel_magnitude),
            np.max(sobel_magnitude)
        ]

        features.extend(edge_features)
        return features

    def preprocess_features(self, use_pca=True, variance_ratio=0.95):
        """特征预处理"""
        print("\n正在进行特征预处理...")

        # 标准化特征
        X_scaled = self.scaler.fit_transform(self.features)

        if use_pca:
            # 保留95%的方差
            self.pca = PCA(n_components=variance_ratio, random_state=42)
            X_pca = self.pca.fit_transform(X_scaled)

            print(f"原始特征维度: {X_scaled.shape[1]}")
            print(f"PCA降维后维度: {X_pca.shape[1]}")
            print(f"累计解释方差: {self.pca.explained_variance_ratio_.sum():.3f}")

            return X_pca
        else:
            return X_scaled

    def train_with_cross_validation(self):
        """
        使用交叉验证训练模型
        """
        print("\n" + "=" * 60)
        print("使用交叉验证训练模型")
        print("=" * 60)

        # 预处理特征
        X = self.preprocess_features(use_pca=True, variance_ratio=0.95)
        y = self.labels

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"数据准备完成:")
        print(f"  训练集大小: {X_train.shape[0]}")
        print(f"  验证集大小: {X_val.shape[0]}")
        print(f"  特征维度: {X_train.shape[1]}")

        # 定义模型
        models = {
            'svm': self.train_svm(X_train, y_train),
            'rf': self.train_random_forest(X_train, y_train),
            'xgb': self.train_xgboost(X_train, y_train),
            'knn': self.train_knn(X_train, y_train)
        }

        # 评估单个模型
        print("\n单个模型在验证集上的表现:")
        for name, model in models.items():
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            print(f"  {name.upper()}: 准确率 = {acc:.4f}")

        # 创建堆叠集成模型
        print("\n训练堆叠集成模型...")
        estimators = [
            ('svm', models['svm']),
            ('rf', models['rf']),
            ('xgb', models['xgb']),
            ('knn', models['knn'])
        ]

        # 使用逻辑回归作为最终估计器
        from sklearn.linear_model import LogisticRegression
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5
        )

        stacking_clf.fit(X_train, y_train)

        # 评估堆叠模型
        y_pred_stack = stacking_clf.predict(X_val)
        acc_stack = accuracy_score(y_val, y_pred_stack)
        print(f"堆叠模型准确率: {acc_stack:.4f}")

        # 显示详细分类报告
        print("\n堆叠模型分类报告:")
        print(classification_report(y_val, y_pred_stack, target_names=self.classes))

        # 混淆矩阵
        cm = confusion_matrix(y_val, y_pred_stack)
        print("混淆矩阵:")
        print(cm)

        self.model = stacking_clf

        # 在整个数据集上重新训练最终模型
        print("\n使用所有数据重新训练最终模型...")
        self.model.fit(X, y)

        # 交叉验证评估
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        print(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return self.model

    def train_svm(self, X_train, y_train):
        """训练SVM模型"""
        print("\n优化SVM参数...")

        svm_param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf']
        }

        svm = SVC(random_state=42, class_weight='balanced', probability=True)

        svm_grid = GridSearchCV(
            svm,
            svm_param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=1,
            verbose=0
        )

        svm_grid.fit(X_train, y_train)
        best_svm = svm_grid.best_estimator_
        print(f"SVM最佳参数: {svm_grid.best_params_}")
        print(f"SVM交叉验证分数: {svm_grid.best_score_:.4f}")

        return best_svm

    def train_random_forest(self, X_train, y_train):
        """训练随机森林模型"""
        print("\n优化随机森林参数...")

        rf_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rf = RandomForestClassifier(random_state=42, class_weight='balanced')

        rf_grid = GridSearchCV(
            rf,
            rf_param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=1,
            verbose=0
        )

        rf_grid.fit(X_train, y_train)
        best_rf = rf_grid.best_estimator_
        print(f"随机森林最佳参数: {rf_grid.best_params_}")
        print(f"随机森林交叉验证分数: {rf_grid.best_score_:.4f}")

        return best_rf

    def train_xgboost(self, X_train, y_train):
        """训练XGBoost模型"""
        print("\n优化XGBoost参数...")

        xgb_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }

        xgb_clf = xgb.XGBClassifier(
            random_state=42,
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False
        )

        xgb_grid = GridSearchCV(
            xgb_clf,
            xgb_param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=1,
            verbose=0
        )

        xgb_grid.fit(X_train, y_train)
        best_xgb = xgb_grid.best_estimator_
        print(f"XGBoost最佳参数: {xgb_grid.best_params_}")
        print(f"XGBoost交叉验证分数: {xgb_grid.best_score_:.4f}")

        return best_xgb

    def train_knn(self, X_train, y_train):
        """训练KNN模型"""
        print("\n优化KNN参数...")

        knn_param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

        knn = KNeighborsClassifier()

        knn_grid = GridSearchCV(
            knn,
            knn_param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=1,
            verbose=0
        )

        knn_grid.fit(X_train, y_train)
        best_knn = knn_grid.best_estimator_
        print(f"KNN最佳参数: {knn_grid.best_params_}")
        print(f"KNN交叉验证分数: {knn_grid.best_score_:.4f}")

        return best_knn

    def save_model(self, filename='optimized_plant_classifier.pkl'):
        """保存模型"""
        if self.model is not None:
            save_data = {
                'model': self.model,
                'scaler': self.scaler,
                'pca': self.pca,
                'classes': self.classes,
                'img_size': self.img_size,
                'label_encoder': self.label_encoder,
                'best_params': self.best_params_
            }

            joblib.dump(save_data, filename)
            print(f"\n模型已保存为 '{filename}'")
            print(f"文件大小: {os.path.getsize(filename)} 字节")
        else:
            print("警告: 没有训练好的模型可以保存")


def main():
    """
    主函数：执行优化植物分类任务
    """
    # 设置数据集路径
    data_path = "/kaggle/input/poject1/dataset-for-task1/dataset-for-task1/train"

    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"错误：路径 {data_path} 不存在！")
        print("请修改为正确的数据集路径")
        return

    print("=" * 70)
    print("机器学习课程设计 - 植物分类系统（特征工程优化版）")
    print("=" * 70)
    print("特征工程:")
    print("  1. 颜色特征 (RGB直方图 + HSV统计)")
    print("  2. HOG特征 (梯度方向直方图)")
    print("  3. LBP特征 (局部二值模式)")
    print("  4. 形状特征 (Hu矩)")
    print("  5. 边缘特征 (Sobel算子)")
    print("模型: SVM + 随机森林 + XGBoost + KNN 堆叠集成")
    print("=" * 70)

    # 创建分类器
    classifier = OptimizedPlantClassifier(data_path, img_size=(64, 64))

    try:
        # 1. 加载所有数据
        classifier.load_all_images()

        # 2. 训练模型
        model = classifier.train_with_cross_validation()

        # 3. 保存模型
        classifier.save_model('optimized_plant_classifier.pkl')

        print("\n" + "=" * 70)
        print("植物分类任务完成！")
        print(f"总图片数: {len(classifier.features)}")
        print(f"特征维度: {classifier.features.shape[1]}")
        print(f"PCA后维度: {classifier.pca.n_components_ if classifier.pca else '未使用'}")
        print(f"模型类型: {type(model).__name__}")
        print(f"模型已保存为 'optimized_plant_classifier.pkl'")
        print("=" * 70)

    except MemoryError:
        print("\n内存不足！尝试以下解决方案:")
        print("1. 进一步减小图像尺寸: 将 img_size=(64, 64) 改为 (32, 32)")
        print("2. 减少特征维度: 修改PCA参数，如使用 variance_ratio=0.90")
        print("3. 关闭其他程序释放内存")
    except Exception as e:
        print(f"\n运行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()