<h1 align="center" id="title">FAI_autonomous-car</h1>

<p align="center"><img src="https://socialify.git.ci/Omar61554/FAI_autonomous-car/image?description=1&amp;font=KoHo&amp;language=1&amp;name=1&amp;owner=1&amp;pattern=Solid&amp;theme=Auto" alt="project-image"></p>

<p id="description">This code is a deep learning model for a driving simulator using a Convolutional Neural Network (CNN). The model is trained to predict the steering angle based on input images.</p>
<h2>Model</h2>

This code is a deep learning model for a driving simulator using a Convolutional Neural Network (CNN). The model is trained to predict the steering angle based on input images.
it defines a custom CNN model called CNNModel by subclassing nn.Module. The model consists of several convolutional layers (conv1, conv2, conv3, conv4) with ReLU activation functions and max pooling operations. There are also fully connected layers (fc1 and fc2) for further feature extraction. Additionally, there is a linear regression branch and two more fully connected layers (fc3 and fc4) to produce the final output.

The code also includes a custom dataset class called CustomDataset, which inherits from the Dataset class provided by PyTorch. This dataset class loads the training data from a CSV file, reads the image paths and labels, and applies transformations if specified. It uses OpenCV (cv2) to read the images and the ToTensor transform to convert them to tensors.

After defining the dataset, the code splits it into training, validation, and test sets using the random_split function from torch.utils.data. Data loaders are created for each set to efficiently load the data during training.

The model is then initialized, and the loss function (MSELoss) and optimizer (Adam) are defined. The code trains the model for a specified number of epochs, iterating over the training data in batches. It calculates the loss, performs backpropagation, and updates the model's parameters using the optimizer.

After training, the model is evaluated on the validation and test sets. The validation loss and test loss are computed using the mean squared error (MSE) loss. Finally, the trained model is saved to a file named "model.pth" using torch.save().

<h2>Results</h2>

![Screenshot (1271)](https://github.com/Omar61554/FAI_autonomous-car/assets/114437079/400b88d9-0964-481c-9917-6829b89e2b07)

![Screenshot (1272)](https://github.com/Omar61554/FAI_autonomous-car/assets/114437079/85387e22-4ca2-421a-aeb1-d93a117873fe)

video link : https://drive.google.com/drive/folders/1ExobRl6ggkKrpd4ZKEXwQIDKoUJ04khT?usp=sharing
<h2>Data</h2>
data from Udacity car simulator: https://drive.google.com/drive/folders/1Gmzv5zfJoyAWZc6FMtn2uVZOHIZmXCa5?usp=sharing

<h2>project on colab</h2>
https://colab.research.google.com/drive/1u2NesD9Rtll165DzzIa1Vvp8xbttKJ3d?usp=sharing
