# Price Prediction Tool (CNN & Regression Model)
## Group Members:

Christina Leung<br>
Ismail Omer<br>
Jacky Zhang<br>
Yug Sharma<br>

# Files:
**Data Folder**: Contains all our raw data, tests files, and Jupyter Notebook Files. Important files to note are:
- EDA_ETL folder contains our EDA and ETL files
- category_colour_cnn folder contains the Category and Colour CNN Jupyter Notebook code
- category_train are the images used to train our category images
- colour_train are the images used to train our colour images
- train_extend are the images used to train our brand images

**clothes_price_prediction_data.csv**: the original clothing CSV file we were training our dataset for to predict the price

# Who and Why
Our tool is designed for general consumers.
When consumers walking on the street and saw someone with good fashion and they want to figure out what it is and how much would be the value.

# How the Project Works
Step1. user give us a picture. </br>
Step2. we use CNN model to predict what it is (e.g. what category/what colour...) </br>
Step3. we use the prediction result, as a test data and use NN model to predict how much price this item should be. </br>
Step4. show the predict price to user.</br>

![Project%20Scope.png](Project%20Scope.png)

# Required Modules
* In order to run the script, please ensure below tools are installed in your local machine:
* AWS CLI (download link: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
* AWS CLI needs to be configured with Access key ID and Access Keys
* `sqlalchemy_utils` and `psycopg2`
* `TensorFlow`,`TorchVision`,`PyTorch`
* `MNIST` database

# Summary of EDA
* In this project, we used a clothes price dataset from Kaggle (Resource: https://www.kaggle.com/datasets/mrsimple07/clothes-price-prediction/data
), and we also use hundreds of online images for model training.
* `Clothes Price Prediction CSV from Kaggle
* `It includes various features related to clothing items along with corresponding prices
* `Features include Brands, Category, Size, Material, Price

  
![Screenshot 2024-02-11 230536](https://github.com/JackyUT2023/project4-group4/assets/139788593/176756e9-9a5b-4382-989a-b6af567c2e10)


![Screenshot 2024-02-11 231426](https://github.com/JackyUT2023/project4-group4/assets/139788593/ce705d40-fb4d-4c33-ae8d-f3421ca3752b)


![Screenshot 2024-02-11 231441](https://github.com/JackyUT2023/project4-group4/assets/139788593/fdb9f0ee-9fc4-4dfc-8aaa-ddc3706efe41)


# Convolutional Neural Network

The CNN component is split up into two models, one for colours which include the colours from our fashion dataset (black, blue, green, red, white, yellow) while the other includes the categories from our dataset (T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, shoes, bag, and ankle boot). We had to make a slight modification to our categories as we decided to use the fashion MNIST dataset for our training data.

The MNIST training data split the clothing images into 10 categories instead of the 6 from our dataset. This will not affect our data as there are more categories in the training set than the test set.

# Colour CNN Model

The NN for the colour model was very simple. It consisted of passing the images into 3 channels (RGB) which helps the image predict the colour much easier. The dataset was compiled by us with over 160 images. It consisted of 3 neurons.


# CNN Colours Analysis
Training the model to learn colours was simpler than categories. The training and validation dataset provided a high accuracy of 80% after 15 epochs.

![colour_accuracy](https://github.com/JackyUT2023/project4-group4/assets/127992819/6956744a-7548-4bb5-955f-069869851271)

The results of the predicted colours are as shown.

![colour_prediction](https://github.com/JackyUT2023/project4-group4/assets/127992819/77839265-1426-4b04-82d4-ebcc7831f1cf)

# CNN Categories Training Model

Training the model to predict clothing classification was much harder than colour as it must learn the shape of the item of clothing as well. Originally, we compiled the images for each category ourselves but we found that the accuracy was much lower with only 50%. This is due to the vast variation of clothing within each category. To reduce variability, we would either have to:

- download more images up to tens of thousands
- decrease the number of categories
- decrease the variability in shapes and colours


We were able to stumble upon the Fashion MNIST image library. Not only was this library large with over 70,000 images and 10 categories, the training images were much more consistent. Each training image was black and white with a resolution of 28x28 pixels. As stated earlier, there were 10 categories instead of 6 that we needed for our data. This, again, would not be an issue as all 6 categories within our clothing_price dataset is a subset of the MNIST categories.

![Fashion-MNIST-dataset](https://github.com/JackyUT2023/project4-group4/assets/127992819/0d91632b-bf32-489e-b35c-0498319d4e4d)

# CNN Categories Analysis

The model ran with 3 neurons and 10 epochs with a runtime of 23 minute runtime (2-3 minute/ epoch). The images were upscaled from 28x28 to 224x224. The images we provided were then downscaled to 224x224 and converted to black and white to match the training dataset as closely as possible. The model ran with an accuracy of 83-84%. We noticed that running with more number of epochs would increase the confidence level of the model. However, with lower number of epochs, each time the model trains, the confidence changes.

![model_accuracy](https://github.com/JackyUT2023/project4-group4/assets/127992819/99de2185-eabf-4548-a5cc-bfc8e10669ac)

Below is an example that shows the probabilities of the same dress with different runs of the model. The title above the article of clothing denotes the actual classification of the clothing while the bar graph on the left is the predicted probabilities:

![dress_prob](https://github.com/JackyUT2023/project4-group4/assets/127992819/a845f2e6-ebe0-492e-9c0e-c5ac7f97e3bf)

![dress_prob2](https://github.com/JackyUT2023/project4-group4/assets/127992819/537439c3-bc2a-4310-b244-7dfad1f7063a)

The model is still able to predict the image shown is a dress. However, the confidence levels are different even if the hyperparameters are the same. The only difference is rerunning the model with a new instance. Despite some slight change sin probabilities, the model can accurately identify items that are similar to the fashion MNIST dataset i.e.) black and white, and shape of the clothing.

![jacket_prob](https://github.com/JackyUT2023/project4-group4/assets/127992819/4c4955af-4a3f-4f18-8846-2b642b799ed4)

# CNN Brand Training Model
- Only 20 training images are used for each brand. </br>
- Fast testing speed and high accuracy for internal resources.

![Screenshot 2024-02-11 231208](https://github.com/JackyUT2023/project4-group4/assets/139788593/cdd20406-1e3f-452c-8dab-ee48dee06569)



# Limitation

There are many factors that we could improve and optimize in our models:

- In attempting to create a machine learning model to predict prices, it was observed that there was no correlation between the price and other features in the dataset. More datasets could provide more insight as to how features affect price
- Premade MNIST training data instead of our own compiled images. This would pose a problem as any variation in an article of clothing that is not similar in orientation, shape, and colour to the MNIST dataset would not be accurately predicted by our model
- Not optimized: run-time 2-3 min/epoch - Upsizing 28x28 to 224x224, increasing runtime 100x
- Low epochs = unable to train all data properly and creates variation in probabilities in different reruns of the model
- Fixed labels/categories/classifications within MNIST training dataset. The model would not be able to predict articles of clothing not within our dataset images such as socks or hats
- Brand training image prediction manually cropped from Internet. (Time consuming to scale it up) 
- AWS Free Tier limitation. (Cloud ETL is limited because only certain request usage of services are free)


# Actionable items to improve the model
* Feeding more training data (e.g. loading more raw images to training database, rotating the current images as new training data etc.) can definatly improve the model.
* Cropping key identical images. (For example, cropping the logo as training data instead of the whole shoe with background could reduce the noise. )


# References
Fashion MNIST dataset: https://www.kaggle.com/datasets/zalando-research/fashionmnist </br>
Clothing dataset: https://www.kaggle.com/datasets/mrsimple07/clothes-price-prediction/data </br>
PyTorch Color Classifier: https://www.kaggle.com/code/kimduhan/cnn-fashion-color-classifier-with-pytorch </br>
TensorFlow MaxPooling2D Layer：https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPooling2D </br>
TensorFlow CONV2D Layer: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html </br>
SQLAlchemy-Utils: https://sqlalchemy-utils.readthedocs.io/en/latest/ </br>
psycopg2: https://pypi.org/project/psycopg2/ </br>
AMS CLI: https://docs.aws.amazon.com/managedservices/latest/onboardingguide/install-cli.html </br>
