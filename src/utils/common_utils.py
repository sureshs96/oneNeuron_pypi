import matplotlib.pyplot as plt
import numpy as np
import joblib # For saving my model as a binary file
from matplotlib.colors import ListedColormap
import os
import logging

def prepare_data(df):
    """it is used to generate the dependent and indenpendent vaariables

    Args:
        df (pd.DataFrame): it is a pandas dataframe

    Returns:
        tuple: returns both dependent and independent variables
    """
    logging.info("Started preparing the data into dependent and independent variables")
    X = df.drop("y", axis =1)
    y = df["y"]
    return X,y

def save_model(model, filename):
    """it is used to save the created model

    Args:
        model (python object): pass the created model
        filename (str): path to dave the model
    """
    logging.info("Started saving the model")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok =True) # only creates if model dir not exists
    filepath = os.path.join(model_dir, filename)
    joblib.dump(model, filepath)
    logging.info("Saved the model")

def save_plot(df, filename, model):
    def _create_base_plot(df):
        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        figure = plt.gcf() # get current figure
        figure.set_size_inches(10, 8)

    def _plot_decision_regions(X, y, classfier, resolution=0.02):
        logging.info("Started plotting the decision regions")
        colors = ("red", "blue", "lightgreen", "gray", "cyan")
        cmap = ListedColormap(colors[: len(np.unique(y))])

        X = X.values # as a array
        x1 = X[:, 0] 
        x2 = X[:, 1]
        x1_min, x1_max = x1.min() -1 , x1.max() + 1
        x2_min, x2_max = x2.min() -1 , x2.max() + 1  

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                            np.arange(x2_min, x2_max, resolution))
        # print(xx1)
        # print("ravel: ",xx1.ravel())
        Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        plt.plot()



    X, y = prepare_data(df)

    _create_base_plot(df)
    _plot_decision_regions(X, y, model)

    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
    plotPath = os.path.join(plot_dir, filename) # model/filename
    plt.savefig(plotPath) 
    logging.info("saved the plot")