# 🧠 Digit Recognition Model & Web App 🚀


This project includes a deep learning model trained for handwritten digit recognition using [dataset](https://www.kaggle.com/datasets/sujaymann/handwritten-english-characters-and-digits) from kaggle. Alongside the model, a web application is provided to test predictions in real-time. There is also all code required to train the model yourself.

You can try the app [here](http://143.47.247.147/).
(note: the VM is pretty slow, so it takes some time to give result:)  )

# Structure
*/web_app*
This file includes everything required for the web app. I am using **node.js** server which communicated with **python** part (which performs recognition) using IPC.

*/model_training*
Here all the code for model training is located. It includes */dataset*,  where the dataset from the link above should be placed (note: the folder structure is changed. The dataset also includes all the letters, which is outside the scope of the project). The *visualize_feature_maps.py* file is used to see how the picture is split into feature maps after convolutions and pooling. */manaul_tests* is where you place your image with digit to be recognised when running *predict.py*.
