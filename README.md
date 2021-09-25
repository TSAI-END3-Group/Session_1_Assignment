# Neural Networks #

This repo answers all the basic questions in and around Neural Networks also known as artificial neural networks (ANNs).

The Colab notebook link for the Assignment is
[Link](https://github.com/TSAI-END3-Group/Session_1_Assignment/blob/main/END3_0_Session_1.ipynb) where we have successfully achieved the desired outcome applying the below conditions:
1. remove the last activation function
2. make sure there are in total 44 parameters
3. run it for 2001 epochs

**NOTE:**
While training this Neural Network for the X-OR gate, we observed that using MSE-Loss over L1-Loss results in much lower error for the network.

#### Model Summary:
1. **Input Layer:** It comprises of 2 neurons as inputs which are connected to hidden layer (which has 5 neurons )
`#params = 5*2+5(Bias) = 15 Parameters`
2. **Two Hidden Layer:** It conatins 5 and 4 neurons respectively.
`#params = 5*4+4(bias) = 24 Parameters`
3. **Output Layer:** Final layer which has a single neuron and connected through hidden layer having 4 neurons.
`#params = 4*1+1(bias) = 5 Parameters`
```Total Parameters = 15+24+5 = 44```

**NOTE**
We also tried training a basic Neural Network model in excel for different learning rates to see how the loss varies w.r.t the learning rates. Please find the below the link for the excel file with the Loss values and Error charts for different Learning rates. [Link](https://github.com/TSAI-END3-Group/Session_1_Assignment/blob/main/Simple_NN_ForwardProp_Backpropagation.xlsx)

## Assignment Questions:

We have answered the following questions in a separate file [Link](https://github.com/TSAI-END3-Group/Session_1_Assignment/blob/main/questions.ipynb)
1. **What is a neural network neuron?**
2. **What is the use of the learning rate?**
3. **How are weights initialized?**
4. **What is "loss" in a neural network?**
5. **What is the "chain rule" in gradient flow?**





## Contributors

* Rohit Agarwal
* Vivek Kumar
* Kushal Gandhi
* Ammishaddai U

