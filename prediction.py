# Starter code for CS 165B HW4
import tensorflow as tf
import numpy as np
import os

"""
Implement the testing procedure here. 

Inputs:
    Unzip the hw4_test.zip and place the folder named "hw4_test" in the same directory of your "prediction.py" file, your "prediction.py" need to give the following required output.

Outputs:
    A file named "prediction.txt":
        * The prediction file must have 10000 lines because the testing dataset has 10000 testing images.
        * Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        * The prediction results must follow the same order of the names of testing images (0.png – 9999.png).
    Notes: 
        1. The teaching staff will run your "prediction.py" to obtain your "prediction.txt" after the competition ends.
        2. The output "prediction.txt" must be the same as the final version you submitted to the CodaLab, 
        otherwise you will be given 0 score for your hw4.

**!!!!!!!!!!Important Notes!!!!!!!!!!**
    To open the folder "hw4_test" or load other related files, 
    please use open('./necessary.file') instead of open('some/randomly/local/directory/necessary.file').

    For instance, in the student Jupyter's local computer, he stores the source code like:
    - /Jupyter/Desktop/cs165B/hw4/prediction.py
    - /Jupyter/Desktop/cs165B/hw4/hw4_test
    If he/she use os.chdir('/Jupyter/Desktop/cs165B/hw4/hw4_test'), this will cause an IO error 
    when the teaching staff run his code under other system environments.
    Instead, he should use os.chdir('./hw4_test').


    If you use your local directory, your code will report an IO error when the teaching staff run your code,
    which will cause 0 score for your hw4.
"""
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
