#This file creates a new model and saves the parameters into a file, which is then loaded into prediction.py
import tensorflow as tf
import pathlib

def model():
    trainDataDir = pathlib.Path('./hw4_train')
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(trainDataDir, validation_split = 0.2, subset = "training", seed = 119, image_size = (28, 28), batch_size = 32)
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(trainDataDir, validation_split = 0.2, subset = "validation", seed = 119, image_size = (28, 28), batch_size = 32)
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    nn = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(28, 28, 3)),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(10)
        ])

    nn.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    nn.fit(train_dataset, validation_data = val_dataset, epochs = 100)

    nn.save("model")
    print("Model Saved")

if __name__ == "__main__":
    model()