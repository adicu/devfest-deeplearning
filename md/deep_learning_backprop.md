
# Backpropagation

Now that we know what *gradient descent* is, and we have an idea of why we would want the gradient of our *loss function* ($C$) - how does one actually go about computing the gradient of a loss function?

The *backprogagation algorithm* is a fast algorithm to compute gradients - without it, none of the neural networks we use today would be able to function efficiently.

As touched on earlier, our neural network learns its weights and biases using gradient descent by minimizing loss. The key concept in backpropagation is the partial derivative of the loss function with respect to any weight $w$ or bias $b$ - $\frac{\partial C}{\partial w}$ or $\frac{\partial C}{\partial b}$ - which gives us an idea of how the loss will change when a bias or weight is changed. This is the power of understanding the backpropagation algorithm - it gives us the intuition necessary to manipulate the behaviour of our neural network by changing weights and biases. Understanding backpropagation gives you the key to open up the "black box" of a neural network.


## The Motivation

Before diving into any of the math, lets try and motivate the *need* for backpropagation.

Our goal is to see how the loss function changes when we change a given weight. Sounds pretty simple, right? Just change the weight in question,  $w_i$, and calculate the loss $C(w)$. 

However, what seems like "just" changing a single weight quickly snowballs into a series of changes that effectively forces us to recalculate all of the weights in our networks. Let's break that down:

1. We change a single weight.
![alt text](http://neuralnetworksanddeeplearning.com/images/tikz22.png)

2. Th

However, changing that single weight changes the output activation of the corresponding neuron, which then causes changes in all of the activations in the next layer, and the next, and so on. So, recalculating this loss function after changing a single weight requires us to pass through the entire neural network again!

To make matters worse, consider how many different weights and biases a single neural network can have - millions! All of a sudden, changing each weight/bias individually and recalculating the loss function each time seems a lot more daunting, right? 

As we'll see, backpropagation allows us to calculate all of the partial derivatives in one pass forward and one pass backward through the neural network. So, instead of calculating the loss function a million times (which requires a million forward passes), we now just need to make a forward and a backward pass. Pretty great!

## The Four Fundamental Equations

below, interpret $j$ and $l$ to mean we are working with $j^{th}$ neuron in layer $l$, out of $L$ layers

also interpret $s \odot t$ to be the *elementwise* product of the vectors s and t

### Equation 1 - error in the output layer

$\delta_j^L=\frac{\partial C}{\partial a_j^L} \sigma ' (z_j^L)$

#### What does this mean?

$\frac{\partial C}{\partial a_j^L}$ - how fast does the loss change depending on the $j^{th}$ output activation (the activation of the output in the last layer)

$\sigma'(z_j^L)$- how fast does the activation function $\sigma$ change at $z_j^L$

This is a component-wise equation - we can convert it to the equivalent matrix-based form easily

$\delta^L = \nabla_a C $ $\odot$ $\sigma'(z^L)$

but we will use the component-wise equation for convenience.


### Equation 2 - error $\delta^l$ in terms of the error in the next layer $\delta^{l+1}$

$\delta^l=((w^{l+1})^T\delta^{l+1})\odot \sigma'(z^l)$

#### What does this mean?

$(w^{l+1})^T$ - the transpose of the weight matrix $w^{l+1}$ for the $l+1^{th}$ layer

We can think of this as moving the error $\delta^{l+1}$ backward to the output of the $l^{th}$ layer by applying the transpose weight matrix, and then backward through the activation function in layer l (by taking the elementwise product $\odot \sigma'(z^l)$) to arrive at $\delta^l$

Thus, we can use Equation 1 to calculate $\delta^L$, and then use Equation 2 to calculate all the other layers' errors by moving backwards through the network.

### Equation 3 - rate of change of cost with respect to bias

$\frac{\partial C}{\partial b^l_j} = \delta^l_j$

#### What does this mean?

The error $\delta^l_j$ is equal to the rate of change of cost with respect to bias $\frac{\partial C}{\partial b^l_j} $

Since given Equation 1 and 2, we can compute any $\delta^l_j$, we can compute any $\frac{\partial C}{\partial b^l_j}$ as well

### Equation 4 - rate of change of cost with respect to weight

$\frac{\partial C}{\partial w^l_{jk}} = a_k^{l-1}\delta_j^l $

or

$\frac{\partial C}{\partial w}=a_{\text{in}}\delta_{\text{out}}$

#### What does this mean?

The rate of change of cost with respect to weight is the product of the activation of the neuron *input* to the weight $w$, and the error of the neuron *output* from the weight $w$.



### Some Useful Intuition
When the sigmoid function is approximately 0 or 1, it is very flat. In Equation 1, this gives us $\sigma'(z_j^L)\approx 0$. Essentially, a weight in the final layer will not change much - will "learn slowly" - if the output neuron is either low () or high () activation (in this case, we call the output neuron *saturated*).

The above logic also applies for the $\sigma'(z^l)$ term in Equation 2, so this intuition can be extended to earlier layers.

Finally in Equation 4, if the activation of a neuron is small $(\approx 0)$, then the gradient term will also be small. Thus the weight will not change much during gradient descent - it will "learn slowly".

## The Backpropagation Algorithm

1. Input x: Set the activation $a^1$ according to the input.

2. Feedforward: For each layer $l = 2, 3, ..., L$ compute $z^l = w^l a^l-1 + b^l$ and $a^l = \sigma(z^l)$

3. Output error $\delta^L$: compute $\delta^L$ (Equation 1)

4. Backpropagate the error: compute  $\delta^l$ for all the earlier layers (Equation 2)

5. Output: The gradient of the cost function is given by Equations 3 and 4.

Step 4 is why the algorithm is called *back*propagation - we compute the error vectors backward, starting from the final layer.

## Additional Resources

The material above is essentially a condensed version of Chapter 2 of Michael Nielsen's wonderful (and free!) [Neural Networks and Deep Learning textbook](http://neuralnetworksanddeeplearning.com/chap2.html). 



For some visual intuition of what's going on in backpropagation, check out 3Blue1Brown's [video](https://www.youtube.com/watch?v=Ilg3gGewQ5U). For a little bit more math, he has a [follow up](https://www.youtube.com/watch?v=tIeHLnjs5U8).




