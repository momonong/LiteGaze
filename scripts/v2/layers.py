import tensorflow as tf
from tensorflow.keras import layers

class CoordinateAttention(layers.Layer):
    """
    Coordinate Attention (CA) Layer
    取代 MobileNetV3 SE-Block 的關鍵組件。
    它能保留 X 和 Y 軸的位置資訊，讓模型學會「看」瞳孔在哪裡。
    """
    def __init__(self, reduction=32, **kwargs):
        super(CoordinateAttention, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        _, h, w, c = input_shape
        mip = max(8, c // self.reduction)

        self.conv1 = layers.Conv2D(mip, (1, 1), use_bias=True)
        self.bn1 = layers.BatchNormalization()
        self.conv_h = layers.Conv2D(c, (1, 1), use_bias=True)
        self.conv_w = layers.Conv2D(c, (1, 1), use_bias=True)
        super(CoordinateAttention, self).build(input_shape)

    def call(self, inputs):
        # 1. Coordinate Pooling
        # H-Pooling: (B, H, 1, C)
        x_pool_h = tf.reduce_mean(inputs, axis=2, keepdims=True)
        # W-Pooling: (B, 1, W, C)
        x_pool_w = tf.reduce_mean(inputs, axis=1, keepdims=True)
        # Transpose W to concat: (B, W, 1, C)
        x_pool_w = tf.transpose(x_pool_w, perm=[0, 2, 1, 3])

        # 2. Concat & Transform
        y = tf.concat([x_pool_h, x_pool_w], axis=1)
        f = self.conv1(y)
        f = self.bn1(f)
        f = f * tf.nn.relu6(f + 3.) / 6.

        # 3. Split & Attention Generation
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]
        
        x_h, x_w = tf.split(f, [h, w], axis=1)
        x_w = tf.transpose(x_w, perm=[0, 2, 1, 3])

        a_h = tf.nn.sigmoid(self.conv_h(x_h))
        a_w = tf.nn.sigmoid(self.conv_w(x_w))

        return inputs * a_h * a_w

class L2CSHead(layers.Layer):
    """
    L2CS-Net Output Head
    輸出兩個東西：
    1. Classification Logits (用來算 CrossEntropy Loss，強迫模型選邊站)
    2. Regression Value (用來算 MSE，取得精確角度)
    """
    def __init__(self, num_bins=90, bin_min=-90, bin_max=90, **kwargs):
        super(L2CSHead, self).__init__(**kwargs)
        self.num_bins = num_bins
        # 建立角度區間 (例如 -90 到 90 度)
        self.bin_centers = tf.constant(
            tf.linspace(float(bin_min), float(bin_max), num_bins), dtype=tf.float32
        )

    def build(self, input_shape):
        self.fc_pitch = layers.Dense(self.num_bins, name='pitch_logits')
        self.fc_yaw = layers.Dense(self.num_bins, name='yaw_logits')
        super(L2CSHead, self).build(input_shape)

    def call(self, inputs):
        # 預測分類分數 (Logits)
        pitch_logits = self.fc_pitch(inputs)
        yaw_logits = self.fc_yaw(inputs)
        
        # 轉成機率
        pitch_probs = layers.Softmax()(pitch_logits)
        yaw_probs = layers.Softmax()(yaw_logits)
        
        # 計算期望值 (Soft-Argmax) -> 變成連續角度
        pred_pitch = tf.reduce_sum(pitch_probs * self.bin_centers, axis=1, keepdims=True)
        pred_yaw = tf.reduce_sum(yaw_probs * self.bin_centers, axis=1, keepdims=True)
        
        # 輸出順序很重要：[pitch_logits, yaw_logits, pred_pitch, pred_yaw]
        # 我們為了方便，直接把連續值接在後面
        return pitch_logits, yaw_logits, pred_pitch, pred_yaw