
# Loss Functions
---
## Motivation
At this point, you should understand how perceptrons can pass information through a neural network. If a network of perceptrons is the performer on stage delivering the flashy results of deep learning, loss functions are the critics in the audience who can evaluate the result with a standard. As we will see in the next section, networks take the feedback produced by loss functions in order to update their weights (_Note_:  This implies that we have talked about weights in the feed forward section) and produce incrementally better output.

## Purpose of a Loss Function
A loss function provides a measure of the accuracy of the network.  We usually  have some function $g(a)$  that takes as input some neural network $a$ and compares the output of $a$ over all input values $x$ to the true label of $x$. When $g \approx 0$ then our network gives very accurate estimates. In this section, we provide two examples of common loss functions and the associated intuition. In the next section on gradient descent, we will see our loss functions allow networks to learn.

## Mean Squared Error (MSE, $L_2$ loss)
$$g(a) = \frac{1}{2n} \sum_{x} \|y(x) - a(x)\|^2$$
where $y(x)$ is the true label of input value $x$ and $n$ is the number of samples. The MSE simple finds the normed difference between the true value and predicted value of an input value as represented by the summand $\|y(x) - a(x)\|^2$. Remember that the output of the neural network and the true labels are **vectors** that correspond to a probability distribution over each possible label. Thus, if our network performs well, we expect that the normed distance between $a(x)$ and $y(x)$ will be close to 0. Taking the summation over all possible $x$ and normalizing over $2n$ gives us the final error value.

## Cross Entropy

$$g(a) = -\frac{1}{n} \sum_x [y(x) \ln a(x)  + (1 - y(x))\ln ( 1- a(x))]$$ 
Cross Entropy loss is one of the most popular loss functions used in modern Deep Learning architectures. At first glance cross entropy loss makes a lot less intuitive sense than MSE – it isn't even clear that this is a proper loss function. We shall see in the next section, however, that the first derivative of Cross Entropy has some nice properties that give it a "good" learning rate. **NOTE:** Cross Entropy, from a high level point of view, computes the difference in information needed to express the true distribution of labels from the predicted distribution of labels. Further we see that
1. Cross Entropy is always greater than 0.
2. When $a(x)$ approaches $y(x)$ Cross Entropy tends to zero.

Note that these two properties are also characteristic of the MSE.

## Conclusion

The important takeaway from this section is to understand the [purpose](#purpose-of-a-loss-function) of the loss function. In the next section we will see how we can find the gradient of a loss function in order to "teach" our neural network. We will then introduce backpropagation, the key idea that enables learning to efficiently and powerfully propagate thorughout all layers of our network!

---

_Notes_: LF really need to be understood in the context of gradient descent so not sure if we should introduce them **before** or **after** @Jessie's section. Conceptually, loss functions as an isolated concept is pretty simple – they are a set of (preferably smooth) functions that evaluate the effectiveness of an algorithms performance on a specific problem. Seems like it'll be a relatively small section
