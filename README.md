## Using QRNN to remove cloud impact in microwave channels

This repository contains code for a study that focusses on cloud correction of microwave humidity channels.
This study is published as "Can machine learning correct microwave humidity radiances for the influence of clouds?" in AMT.


### Dependencies

For running ARTS simulations and retrieval, we are using the [parts](https://github.com/simonpf/parts) package.
It's is not available on PyPI yet, so you have to install it manually.

````
git clone https://github.com/simonpf/parts
cd parts
pip install -e
````
 
QRNN is available as part of typhon package. 

https://github.com/atmtools/typhon.git


### Contents of the repository

### Simulations 

contains the scripts to run ARTS simulations and retreival

### MWHS

contains the notebooks used to train with MWHS data and scripts to analyse the various results presented in the article.

### ICI 

contains the notebooks used to train with ICI data and scripts to analyse the various results presented in the article.

### SMS

contains the notebooks used to train with SMS data and scripts to analyse the various results presented in the article.

The input data can be provided on request.


