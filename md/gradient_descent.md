
## Gradient Descent


Most deep learning algorithms involve some sort of optimization. We'd like to find the value of  $x$ where the _loss function_ $f(x)$ is minimized, so the output of our network is approximately the output of the function we are representing.

Gradient descent is the most popular method for optimization, because it works with most functions and is relatively easy to implement. The name gradient descent gives away what the method is - "descending" by moving in the direction of most negative slope. The gradient can be thought of as the derivative of a function (but it is a vector field).

Gradient descent is a way to find a minimum point of a function by starting at a point on the function and making many small moves towards a better (smaller) point. Each of these moves is along the direction of most negative gradient, which gets us to the smallest point possible from our starting point. However, we aren't looking for what the minimum is, but what point the minimum occurs at.

Conceptually, one way to think about gradient descent is to think about walking downhill. In this case, you can only take steps of a fixed size. Additionally, every step you take must be in the direction where the slope is steepest.

![](http://www.deepideas.net/wp-content/uploads/2017/08/gradient_descent_2.png)

Mathematically, gradient descent looks like this:
$x' = x - \epsilon \nabla_x f(x)$. $x'$ is the new x value, $x$ is the starting x value, $\nabla_xf(x)$ is the gradient with respect to $x$, and $\epsilon$ is the _learning rate_. Notation may vary, depending who is using it, but this concept is the same. We'll run through a simple example by hand below.

Let's choose the function $y = (x-2)^2$, starting at the point $x_0 = 4$, with a learning rate of 0.01. The derivative is $\frac{dy}{dx}= 2(x-2)$. If you graph this function, you can see that the minimum is located at $x = 2$.

On the first iteration of gradient descent:

$x_1 = x_0 - \epsilon \nabla_x f(x)$

$x_1 = 4 - 0.01(2(4 - 2)) = 3.96$

Second iteration of gradient descent:

$x_2 = x_1 - \epsilon \nabla_x f(x)$

$x_2 = 3.96 - 0.01(2(3.96 - 2)) = 3.92$

Third iteration of gradient descent:

$x_3 = x_2 - \epsilon \nabla_x f(x)$

$x_3 = 3.92 - 0.01(2(3.92 - 2)) = 3.88$

If we started at x = 0, this is what would happen:
On the first iteration of gradient descent:

$x_1 = x_0 - \epsilon \nabla_x f(x)$

$x_1 = 0 - 0.01(2(0 - 2)) = 0.04$

Second iteration of gradient descent:

$x_2 = x_1 - \epsilon \nabla_x f(x)$

$x_2 = 0.04 - 0.01(2(0.04 - 2)) = 0.079$

Third iteration of gradient descent:

$x_3 = x_2 - \epsilon \nabla_x f(x)$

$x_3 = 0.079 - 0.01(2(0.079 - 2)) = 0.117$

We can see that with either start point, with each iteration of gradient descent we get closer to $x=2$, the point of global minimum. We can stop performing iterations when the difference between consecutive iterations is less than  a value that we set, for example 0.000001.

Experiment with the parameters (which x to start at, learning rate, max iterations, precision) in the code below to see gradient descent in action.


```python
# SOURCE: wikipedia
# From calculation, it is expected that the local minimum occurs at x=9/4

cur_x = 4
precision = 0.0001
learning_rate = 0.01
previous_step_size = 1 
max_iters = 1000 # maximum number of iterations
iters = 0 #iteration counter

df = lambda x: 2*(x-2) # our function's derivative

while previous_step_size > precision and iters < max_iters:
    print('iter:' + str(iters) + ' x:' + str(cur_x))
    prev_x = cur_x
    cur_x -= learning_rate * df(prev_x)
    previous_step_size = abs(cur_x - prev_x)
    iters+=1

print("The local minimum occurs at", cur_x)
```

### Choosing a learning rate
![](https://cdn-images-1.medium.com/max/1600/0*QwE8M4MupSdqA3M4.png)
Choosing the right learning rate is something to experiment with when doing gradient descent. With a too large learning rate, there is a chance that you'll step right past the minimum point and bounce around, missing the minimum and even ending up at a point with higher loss. With a too small learning rate, you might end up running many iterations and gradient descent will take forever.

![](https://cdn-images-1.medium.com/max/1600/1*rcmvCjQvsxrJi8Y4HpGcCw.png)

The ideal learning rate will minimize the loss function in the fewest number of iterations necessary. In practice, you just want the loss function to decrease after every iteration. Finding this rate is a matter of expermentation and depends on the project, though common learning rates range from 0.0001 up to 1.

### Why do gradient descent?
But wait - haven't we found the x value that gives the minimum value of a function in calculus? Why even bother with gradient descent?

Long story short, finding the solutions to functions in higher dimensions (functions in terms of multiple variables) can often be very complicated. In these cases, gradient descent is a computationally faster way to find a solution.

Keep in mind that gradient descent doesn't always find the very best solution - the minimum it finds may not be the global minimum.

![](https://www.superdatascience.com/wp-content/uploads/2018/09/Artificial_Neural_Networks_ANN_Stochastic_Gradient_Descent_Img2.png)

### Application in feedforward nets
A network learns a function when its output is accurate to the output of a function. To do this, we use gradient descent to minimize the error of the network's output, by calculating the derivative of the error function with respect to the network weights and changing the weights such that the error decreases.

#### Image sources
http://www.deepideas.net/deep-learning-from-scratch-iv-gradient-descent-and-backpropagation/

https://www.superdatascience.com/wp-content/uploads/2018/09/Artificial_Neural_Networks_ANN_Stochastic_Gradient_Descent_Img2.png

https://cdn-images1.medium.com/max/1600/1*rcmvCjQvsxrJi8Y4HpGcCw.png

https://cdn-images-1.medium.com/max/1600/0*QwE8M4MupSdqA3M4.png

