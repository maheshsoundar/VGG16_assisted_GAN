This is to demonstrate how to train a Generative advererial network on our own dataset and generate new images. GANs generall require a lot of computation resources and data which is not possible at personal level. But the code demonstrates how it  works inernally. VGG 16 feaures have been added discriminator to help it learn faster and more accurate. 

1.Set up virtual env from root directory using python -m venv .venv. Then activate venv by running the activate.bat file in .venv/Scripts.
2.Run pip install -r requirements.txt and make sure all dependencies are installed in .venv/Scripts folder.
3.Make sure the data from https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer (all_data.csv) is downloaded and is available in the data folder before running the scripts. During testing only 'happy' folder from kaggle dataset was used. But any other dataset can be added to that folder. Or the model training will fail. 
4.Run main.py using "python main.py" from a terminal (root directory of repo)

The generated images can be further improved by using more epochs, latent dimensions and of corse more diverse data. But the progress of images generated from noise to something can be seen during the training. The stochastic nature of the algorithm can yield different results at different times. SO try using different random seeds and make a log of results to choose the best. 