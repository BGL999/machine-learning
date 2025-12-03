import os
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
import joblib
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class MultiFeaturePredictor:
    def __init__(self, model_path, test_data_path, img_size=(128, 128)):
        """
        初始化多特征预测器
        Args:
            model_path: 训练好的模型文件路径
            test_data_path: 测试集图像文件夹路径
            img_size: 图像尺寸（必须与训练时一致）
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.img_size = img_size

        # 加载模型
        self.load_model()

        # 检查测试集路径
        if not os.path.exists(test_data_path):
            raise ValueError(f"测试集路径不存在: {test_data_path}")

    def load_model(self):
        """加载训练好的模型和相关参数"""
        print(f"正在加载模型: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 加载保存的模型
        saved_data = joblib.load(self.model_path)

        # 提取模型和预处理组件
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.pca = saved_data['pca']
        self.classes = saved_data['classes']
        self.training_img_size = saved_data.get('img_size', (128, 128))

        # 确保图像尺寸一致
        if self.training_img_size != self.img_size:
            print(f"警告: 训练时图像尺寸为 {self.training_img_size}，预测时设置为 {self.img_size}")
            # 如果不同，使用训练时的尺寸
            self.img_size = self.training_img_size

        print(f"模型加载成功!")
        print(f"类别数量: {len(self.classes)}")
        print(f"类别名称: {self.classes}")
        print(f"图像尺寸: {self.img_size}")
        print(f"特征类型: {saved_data.get('feature_types', ['未知'])}")

    def extract_multi_features(self, img_path):
        """
        提取多种特征（必须与训练时完全一致）
        Args:
            img_path: 图像文件路径
        Returns:
            融合特征向量
        """
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图像 {img_path}")
                return None

            # 调整图像大小
            img_resized = cv2.resize(img, self.img_size)
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            feature_vector = []

            # 1. HOG特征
            hog_features = hog(
                gray,
                orientations=9,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                feature_vector=True
            )
            feature_vector.extend(hog_features)

            # 2. 颜色直方图特征
            # RGB颜色直方图
            for i in range(3):
                hist = cv2.calcHist([img_resized], [i], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                feature_vector.extend(hist)

            # HSV颜色空间
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            for i in range(3):
                if i == 0:  # Hue通道
                    hist = cv2.calcHist([hsv], [i], None, [32], [0, 180])
                else:
                    hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                feature_vector.extend(hist)

            # 3. LBP特征
            radius = 2
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)
            feature_vector.extend(hist)

            # 4. Hu矩
            moments = cv2.moments(gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            feature_vector.extend(hu_moments)

            # 5. 边缘特征统计
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            edge_features = [
                np.mean(sobel_magnitude),
                np.std(sobel_magnitude),
                np.max(sobel_magnitude),
                np.sum(sobel_magnitude > np.mean(sobel_magnitude)) / sobel_magnitude.size
            ]
            feature_vector.extend(edge_features)

            return np.array(feature_vector)

        except Exception as e:
            print(f"提取特征时出错 ({img_path}): {e}")
            return None

    def preprocess_features(self, features):
        """
        特征预处理（与训练时一致）
        Args:
            features: 原始特征
        Returns:
            预处理后的特征
        """
        # 标准化
        features_reshaped = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features_reshaped)

        # PCA降维
        if self.pca is not None:
            features_pca = self.pca.transform(features_scaled)
            return features_pca
        else:
            return features_scaled

    def predict_image(self, img_path):
        """
        对单张图像进行预测
        Args:
            img_path: 图像文件路径
        Returns:
            (预测类别, 预测概率)
        """
        # 提取特征
        features = self.extract_multi_features(img_path)
        if features is None:
            return None, None

        # 预处理特征
        processed_features = self.preprocess_features(features)

        # 预测
        try:
            # 预测类别
            prediction = self.model.predict(processed_features)[0]

            # 获取预测概率（如果模型支持）
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_features)[0]
                confidence = probabilities[prediction]
            else:
                # 对于SVC，需要使用decision_function
                decision_values = self.model.decision_function(processed_features)[0]
                confidence = np.max(decision_values) / 10.0  # 简化处理
                confidence = min(max(confidence, 0.0), 1.0)

            # 获取类别名称
            predicted_class = self.classes[prediction]

            return predicted_class, confidence

        except Exception as e:
            print(f"预测图像时出错 ({img_path}): {e}")
            return None, None

    def predict_test_set(self, output_csv="submission.csv"):
        """
        预测整个测试集并生成CSV文件
        Args:
            output_csv: 输出的CSV文件名
        """
        print(f"\n开始处理测试集: {self.test_data_path}")

        # 获取测试集所有图像文件
        test_images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            import glob
            test_images.extend(glob.glob(os.path.join(self.test_data_path, ext)))

        # 按文件名排序，确保顺序一致
        test_images = sorted(test_images)

        if not test_images:
            print(f"警告: 在 {self.test_data_path} 中没有找到图像文件")
            return

        print(f"找到 {len(test_images)} 个测试图像")

        # 存储结果
        results = []

        # 逐个处理图像
        for i, img_path in enumerate(test_images):
            # 获取文件名（不包含路径）
            filename = os.path.basename(img_path)

            if (i + 1) % 10 == 0:
                print(f"处理图像 {i + 1}/{len(test_images)}: {filename}")

            # 预测
            predicted_class, confidence = self.predict_image(img_path)

            if predicted_class is None:
                print(f"  警告: 无法预测 {filename}，跳过此图像")
                continue

            # 添加到结果列表
            results.append({
                'ID': filename,
                'Category': predicted_class,
                'Confidence': confidence  # 可选：保留置信度信息
            })

        # 创建DataFrame
        df = pd.DataFrame(results)

        # 保存为CSV文件（只保留ID和Category两列）
        submission_df = df[['ID', 'Category']]
        submission_df.to_csv(output_csv, index=False)

        print(f"\n预测完成!")
        print(f"结果已保存到: {output_csv}")

        # 显示统计信息
        self.show_statistics(df)

        return df

    def show_statistics(self, df):
        """显示预测结果的统计信息"""
        print(f"\n{'=' * 60}")
        print("预测结果统计:")
        print('=' * 60)

        # 统计每个类别的数量
        category_counts = df['Category'].value_counts()

        print("\n类别分布:")
        for category, count in category_counts.items():
            percentage = count / len(df) * 100
            print(f"  {category}: {count} 张图片 ({percentage:.1f}%)")

        # 显示前几个结果
        print(f"\n前10个预测结果:")
        print(df.head(10).to_string(index=False))

        # 如果有置信度信息，计算平均置信度
        if 'Confidence' in df.columns:
            avg_confidence = df['Confidence'].mean()
            print(f"\n平均预测置信度: {avg_confidence:.3f}")


def main():
    """
    主函数：执行测试集预测
    """
    # 设置路径
    model_path = "final_multi_feature_classifier.pkl"  # 训练好的多特征模型文件
    test_data_path = '/kaggle/input/poject1/dataset-for-task1/dataset-for-task1/test'
    output_csv = "submission-for-task1.csv"  # 输出的CSV文件名

    print("=" * 70)
    print("植物分类测试集预测系统 - 多特征融合")
    print("=" * 70)

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请确保已经训练并保存了模型")
        return

    # 创建预测器
    try:
        predictor = MultiFeaturePredictor(
            model_path=model_path,
            test_data_path=test_data_path,
            img_size=(128, 128)
        )

        # 预测整个测试集
        results_df = predictor.predict_test_set(output_csv)

        print("\n" + "=" * 70)
        print("测试集预测完成!")
        print(f"结果文件: {output_csv}")
        print("=" * 70)

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()