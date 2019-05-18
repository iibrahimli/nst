# Neural Style Transfer


The idea behind neural style transfer is actually not that hard to grasp if you are already familiar with convolutional neural networks. It was initially presented in a 2016 paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys et al. There are many implementations and tutorials available online. This code is based on the nice Colab [notebook](https://colab.research.google.com/github/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb) by Google. If you wish to try this and do not own a GPU, using Colab or other cloud services is recommended, since using a CPU is very slow (I couldn't even wait for 1000 iterations to finish on my monstrous Core i3).


## Theory

As the researchers mention in the paper, the key takeaway is that the information about the content and style of an image is disentangled and we can separate them by extracting specific features from an image using a CNN. The original paper uses VGG-19 trained on the ImageNet dataset for this purpose. I will give a brief high-level overview here. We consider 3 images: the input image, a content image, and a style image. The input image is the one we modify to obtain the result. Content and style images provide the content and style features respectively. The problem is formulated as an optimization problem, where we aim to minimize a loss function with respect to the input image. The loss function is a linear combination of content loss and style loss (total variation loss may also be added). We change the input image using the gradient from the loss function.

### Content loss

Content loss measures the "difference" between the input and content images. We can take the squared-error loss between images, which is simply the sum of squeared pixel-wise differences:

![Content Loss](rsc/content_loss.png)

Minimising this loss alone would result in a recontruction of the content image.

### Style loss

Unlike the content loss, the style loss captures the differences that are not spatially fixed, e.g., if there is a certain feature in, say, upper right corner of the style image, that feature does not necessarily have to present in the same part of the input image. 