# Description:
This web app allows users to upload images of the fruit or leaf of one of our common compatible plants, and our neural net identifies what type of disease (if any) the plant has. It then displays all relevant info for that disease. If the image uploaded is not a compatible plant leaf or fruit, our app will instead show an epic Bernie x Psy meme.

## Setup Instructions:
To run our app, clone our repository. Then activate a virtual environment with Python verison 3.5 to 3.8 (necessary for compatibility with latest version of Tensorflow). Next cd into Final and run "pip install -r 'requirements.txt'" (pip3 on Mac) to install the required libraries in the virtual environment. Our model is too large to upload to Github, but you can download a pre trained model into the Final/static folder from the Google Drive link here: https://drive.google.com/file/d/1cv09M407sOdQDwhL5ryXoMjRMYlGkMu8/view?usp=sharing. This model is trained on 15 epochs and must be named '15epoch.h5' for the code to work. Lastly, run python app.py, and upload images to your heart's content.

### Creators and Credits: 
This project was created by Rayhan Ahmed, Ashlan Ahmed, David Liu, and Christy Jestin at BoilerMake VIII on Jan. 23-25. We used the Identification of Plant Leaf Diseases dataset from Mendeley Data. It can be found here: https://data.mendeley.com/datasets/tywbtsjrjv/1. A demo of our product can be found here: https://youtu.be/pK9nTkgKxoQ. The name krishi comes from the Sanskrit word for agriculture.
