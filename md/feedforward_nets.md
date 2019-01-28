
## Feedforward networks
Now that we have covered perceptrons and activation functions, we can put them together to create _feedforward neural networks_.

![](https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/Architecture/images/feedforward.jpg)

These networks are important in deep learning models to approximate functions. They're called "feedforward" because information is passed forward through the network and is not passed backwards. (When information is also passed backwards, the network is called a recurrent neural network).

In deep learning specifically, feedforward networks consist of many layers. We'll go through the structure of these networks in this section.

### Structure
Feedforward networks (FFN) consist of _layers_ of perceptrons. Perceptrons are commonly called nodes in the context of neural networks.The simplest feedforward net consists of a single layer of nodes.

 All nodes in a _layer_ are connected to every other node in the previous layer. The _input layer_ is the input to the neural net (short for network). For example, say we have a network that determines if a 32 x 32 picture is a dog. The input layer could be 32x32 = 1,024 input nodes, one node for each pixel in the image. The output layer could contain a single node, and the value of the output would encode whether or not the picture is a dog. _Hidden layers_ are all layers in between the _input layer_ and the _output layer_. 
 
The outputs of nodes in a layer become the inputs to nodes the next layer. In figures, this is often represented by a line which connects one node to nodes in the next layer. Information (data) is passed through each layer of nodes until it gets output through the output layer.

The following figure illustrates a single node in a network. All the inputs to the node are weighted, then summed. This sum is passed through an activation function to keep output within a certain range, then output. If this node is in a hidden layer, this output is then passed to nodes in the next layer as input. If this node is in an output layer, this output is simply output.
![](https://skymind.ai/images/wiki/perceptron_node.png)
It's important to keep in mind that different inputs are weighted differently. A neural network "learns" by adjusting the weights of inputs to each node so that eventual output better approximates a function.

In the image below, there are three inputs (shown as hollow circles), which are passed into a hidden layer with 4 nodes, and the outputs of that layer are passed as inputs into the final output layer, with 5 nodes. We would say this network has two layers.
 ![](http://www.fon.hum.uva.nl/praat/manual/Feedforward_neural_networks_1__What_is_a_feedforward_ne_1.png)
 



### Why have layers?

Layers in neural networks are what make the learing in deep learning "deep". For the complex functions that deep learning tries to approximate, layers are necessary. In fact, the [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) states that a feedforward neural network with a single hidden layer and finite number of neurons can approximate almost any function. Adding hidden layers can improve accuracy, up until a point. Choosing the number of layers often comes down to trial and error.



### How do we get the weights?

The weights in feedforward neural nets are what influence the output. Weights are usually randomly initialized, and then the  _backpropogation algorithm_ is used to update weights from one iteration to the next. We'll talk about this in a later section.



### Other resources

We like 3Blue1Brown's [video explanation](https://www.youtube.com/watch?v=aircAruvnKk&t=0s&index=2&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) of neural nets.

#### Image sources
https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/Architecture/images/feedforward.jpg

https://skymind.ai/images/wiki/perceptron_node.png

http://www.fon.hum.uva.nl/praat/manual/Feedforward_neural_networks_1__What_is_a_feedforward_ne.html

# Exercise
1. Implement a 2-layer neural network. Take the existing neural network class that you created 
