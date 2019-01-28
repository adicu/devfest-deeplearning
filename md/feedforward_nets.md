
# Feedforward networks
Now that we have covered perceptrons and activation functions, we can put them together to create _feedforward neural networks_.

![](https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/Architecture/images/feedforward.jpg)

These networks are important in deep learning models to approximate functions. They're called "feedforward" because information is passed forward through the network and is not passed backwards. (When information is also passed backwards, the network is called a recurrent neural network).

### Structure
Feedforward nets (short for networks) consist of _layers_ of perceptrons. Perceptrons are commonly called nodes in the context of neural networks.The simplest feedforward net consists of a single layer of nodes.

 All nodes in a _layer_ are connected to every other node in the previous layer, though different connections may have different weights. Information (data) is passed through each layer of nodes until it gets output through the output layer. _Hidden layers_ are all layers before the _output layer_. 
 
 ![](http://www.fon.hum.uva.nl/praat/manual/Feedforward_neural_networks_1__What_is_a_feedforward_ne_1.png)
 
 In the image above, there are three inputs (shown as hollow circles), which are passed into a hidden layer with 4 nodes, and the outputs of that layer are passed as inputs into the final output layer, with 5 nodes. Each connection, additionally, has its own weight. We would say this network has two layers.
 
#### Image source
http://www.fon.hum.uva.nl/praat/manual/Feedforward_neural_networks_1__What_is_a_feedforward_ne.html
