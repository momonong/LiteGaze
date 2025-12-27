import tensorflow as tf
import numpy as np
import cv2

def load_data(npz_path):
    """ 載入之前的資料 """
    data = np.load(npz_path)
    images = data['images'] # 假設是 (N, 60, 60, 1)
    labels = data['labels'] # (N, 2) [Pitch, Yaw]
    return images, labels

def augment_image(image, label):
    """
    Data Augmentation: 模擬 Webcam 的爛畫質
    """
    # 1. 隨機亮度/對比度 (模擬光線變化)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # 2. 隨機雜訊 (模擬 Webcam 噪點)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
    image = tf.add(image, noise)
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    # 3. 確保是 60x60
    image = tf.image.resize(image, (60, 60))
    
    return image, label

def get_dataset(images, labels, batch_size=32, is_train=True):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if is_train:
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# === L2CS 的特殊標籤轉換 ===
def convert_label_to_bins(label, num_bins=90, min_angle=-90, max_angle=90):
    # 將連續角度轉成 Bin 的 Index (分類標籤)
    # label: (Pitch, Yaw) in degrees
    bins = np.linspace(min_angle, max_angle, num_bins)
    bin_idx = np.digitize(label, bins) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)
    return bin_idx