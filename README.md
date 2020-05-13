# Code of our FYP

This repository contains the complete implementation of my final year project, details of the contents can be found in my dissertation (later will be published to arXiv)

## Dependencies
- Python >= 3.6.5
- Pytorch 1.4.0
- Gensim  3.8.1
- emoji 0.5.4
- nltk 3.4.5

## Usage ##
The code has been tested on MacOS, Windows and Ubuntu. For mac users, there is a known bug caused by the multiprocess package of Python, you can ignore the warning message since it doesn't affect the output. The scripts are in the 'code' file, and they are independent from each other. un_data and su_data directories in 'code' contain our own datasets, while in 'cluster', you can find 6 more open-sourced short text datasets.

**1. Preprocess data**   
  	 We provide a python script to preprocess data

    $ python preparedata.py <fname> <outpath>
    fname    the path of the file that you want to process
    outpath  the path of where you want to save the result

**2. Train a classifier**  
   We provide a pre-trained TextCNN used for data filtering, you can train your own model by run:   

    $ python training.py <embed> <train> <test> <dataset> <epoch> <restart> <savestep>
      embed     the path of pre-trained word embeddings
      train     the path of training set
      test      the path of test set
      dataset   the name of the dataset
      epoch     the maximum training epoch
      restart   whether to continue training a existing model
      savestep  when to save the model
 
   The trained model will be saved to current directory with name `$dataset$_model.pkl` by default

**3. Filtering data**     
   We provide a filtered data for demo, but you can use your own model and your dataset by run

    $ python filtering.py <fname> <dataset>
    fname   the path of the data needed to be filtered
    dataset the name of the dataset
    
    
   By default, the script will load the model from <dataset>_model.pkl file in the current directory
   
   The result will be output to '$dataset$.txt' file in 'temp' directory
  
**4. Topic discovery and visualization**    
   Finally, we provide a python script to generate topics of given corpus.
  
    $ python clustering.py <fname> <model> <iter> <K> <maxwords> <dataset>
      fname    the path of the file needed to be clustered
      model    which model you want to use, currently we only provide  'ebtm' and 'btm'
      iter     the maximum iteration of training
      K        the number of topics you want to generate
      maxwords the number of words that used to represent a topic
      dataset  the name of the given corpus
  
 The output will be saved to the 'visulization' directory with prefix '$dataset$'
      

## Related codes ##
- [BTM](https://github.com/xiaohuiyan/BTM)


If there is any question, feel free to contact: [Yuyang Liu](zy22049@nottingham.edu.cn).
