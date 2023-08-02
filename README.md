# oneNeuron_pypi
oneNeuron python package creation

## How to use -
```python
    from oneNeuron.perceptron import Perceptron
    from utils.common_utils import prepare_data, save_model, save_plot
    ## Get X and y from dataframe
    X,y = prepare_data(df)
    model = Perceptron(eta, epochs)
    model.fit(X,y)
    _ = model.total_loss()
    save_model(model, modelname)
    save_plot(df , plotfilename, model)
```
# Reference -
[official python docs](https://packaging.python.org/tutorials/packaging-projects/)

[github docs for github actions](https://docs.github.com/en/actions/guides/building-and-testing-python#publishing-to-package-registeries)