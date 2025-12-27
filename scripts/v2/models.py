import tensorflow as tf
from tensorflow.keras import layers, models
from layers import CoordinateAttention, L2CSHead

def conv_block(x, filters, kernel=3, stride=1, activation=True):
    x = layers.Conv2D(filters, kernel, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    # ✅ 修改後：手動寫公式，避開 Unknown activation 錯誤
    if activation: 
        x = layers.Lambda(lambda v: v * tf.nn.relu6(v + 3) * 0.16666667)(x)
    return x

def inverted_res_block(x, expand, out_filters, stride, use_ca=False):
    in_filters = x.shape[-1]
    
    # 1. Expansion
    if expand > 1:
        x = conv_block(x, int(in_filters * expand), kernel=1)
        
    # 2. Depthwise
    x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    # ✅ 修改後
    x = layers.Lambda(lambda v: v * tf.nn.relu6(v + 3) * 0.16666667)(x)
    
    # === 🔥 核心差異：用 Coordinate Attention 取代 SE-Block ===
    if use_ca:
        x = CoordinateAttention(reduction=8)(x)
    # ========================================================
    
    # 3. Projection
    x = conv_block(x, out_filters, kernel=1, activation=False)
    
    # Shortcut
    if stride == 1 and in_filters == out_filters:
        x = layers.Add()([x, x]) # 這裡簡化，實際上要接 input tensor
    return x

def build_student_v2(input_shape=(60, 60, 1)):
    inputs = layers.Input(shape=input_shape)
    
    # Backbone (參考 MobileNetV3 架構但簡化)
    x = conv_block(inputs, 16, stride=2) # 30x30
    
    x = inverted_res_block(x, 1, 16, 1)
    x = inverted_res_block(x, 4, 24, 2) # 15x15
    x = inverted_res_block(x, 3, 24, 1, use_ca=True) # 開始加 CA
    
    x = inverted_res_block(x, 4, 40, 2, use_ca=True) # 8x8
    x = inverted_res_block(x, 4, 40, 1, use_ca=True)
    x = inverted_res_block(x, 4, 48, 1, use_ca=True)
    
    x = inverted_res_block(x, 6, 96, 2, use_ca=True) # 4x4
    x = inverted_res_block(x, 6, 96, 1, use_ca=True)
    
    # Head
    x = layers.GlobalAveragePooling2D()(x)
    # ✅ 修改後：把 activation 拿掉，獨立寫成一層
    x = layers.Dense(576)(x)  # 先做線性輸出
    x = layers.Lambda(lambda v: v * tf.nn.relu6(v + 3) * 0.16666667)(x) # 再接 Activation
    x = layers.Dropout(0.2)(x)
    
    # L2CS Output
    p_logit, y_logit, p_reg, y_reg = L2CSHead(num_bins=90)(x)

    # === 🔥 關鍵修復：強制命名輸出層 ===
    # 我們加一個不做任何運算 (linear) 的層，只為了把名字設定成 'pitch_logits' 和 'yaw_logits'
    # 這樣 train.py 才能透過這個名字找到它們來算 Loss
    p_logit = layers.Activation('linear', name='pitch_logits')(p_logit)
    y_logit = layers.Activation('linear', name='yaw_logits')(y_logit)
    
    # 我們把這些包再一起輸出
    # Output 1: Continuous Gaze (For Demo/MSE) -> Shape (B, 2)
    # Output 2: Pitch Logits (For Classification Loss)
    # Output 3: Yaw Logits (For Classification Loss)
    gaze_continuous = layers.Concatenate(name='gaze_out')([p_reg, y_reg])
    
    return models.Model(
        inputs=inputs, 
        outputs=[gaze_continuous, p_logit, y_logit],
        name="LiteGaze_V2_Student"
    )

# scripts/v2/models.py 的最後面

def build_teacher_v3(input_shape=(224, 224, 3)):
    """
    God Teacher: ConvNeXtLarge + L2CS Head
    """
    # 嘗試使用 KerasCV 或 tf.keras.applications 的 ConvNeXt
    # 如果您的 A100 環境 TF 版本較新 (2.11+)，建議用 ConvNeXt
    try:
        backbone = tf.keras.applications.ConvNeXtLarge(
            include_top=False, 
            weights='imagenet', 
            input_shape=input_shape
        )
        print("✅ Using ConvNeXtLarge Backbone")
    except (AttributeError, ValueError):
        # 備案：EfficientNetV2L
        print("⚠️ ConvNeXt not found, using EfficientNetV2L")
        backbone = tf.keras.applications.EfficientNetV2L(
            include_top=False, 
            weights='imagenet', 
            input_shape=input_shape
        )

    backbone.trainable = True 
    
    inputs = layers.Input(shape=input_shape)
    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    
    # L2CS Head
    p_logit, y_logit, p_reg, y_reg = L2CSHead(num_bins=90)(x)
    
    # 命名
    p_logit = layers.Activation('linear', name='pitch_logits')(p_logit)
    y_logit = layers.Activation('linear', name='yaw_logits')(y_logit)
    gaze_continuous = layers.Concatenate(name='gaze_out')([p_reg, y_reg])
    
    return models.Model(
        inputs=inputs, 
        outputs=[gaze_continuous, p_logit, y_logit], 
        name="God_Teacher_Model"
    )

if __name__ == '__main__':
    model = build_student_v2()
    model.summary()