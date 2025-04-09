import tensorflow as tf

class CustomAttentionLayer(Layer):
    def __init__(self, heads=4, key_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.key_dim = key_dim
    
    def build(self, input_shape):
        self.query_dense = Dense(self.heads * self.key_dim)
        self.key_dense = Dense(self.heads * self.key_dim)
        self.value_dense = Dense(self.heads * self.key_dim)
        self.output_dense = Dense(input_shape[-1])
        super().build(input_shape)
    
    def call(self, inputs):
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attended = tf.matmul(attention_weights, value)
        return self.output_dense(attended)

def create_advanced_mortality_model(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    attention_output = CustomAttentionLayer(heads=8, key_dim=64)(x)
    combined = x + attention_output
    x = Dense(64, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    return model