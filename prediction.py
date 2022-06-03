# Starter code for CS 165B HW4
import tensorflow as tf
import numpy as np
import os

def prediction():
    nn = tf.keras.models.load_model("model")

    print('Running predictions\n')
    file = open("prediction.txt", 'w')
    for i in range(0,10000):
        image = str(i) + '.png'
        print("Predicting " + str(image))
        image_file = os.path.join('./hw4_test', image)
        img = tf.keras.utils.load_img(image_file)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = nn.predict(img_array)
        labels = tf.nn.softmax(predictions[0])
        file.write(str(np.argmax(labels)))
        if i < 9999:
            file.write("\n")
            i = i + 1
    file.close()
    print('Done')

if __name__ == "__main__":
    prediction()
