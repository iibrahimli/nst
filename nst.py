import numpy as np
from PIL import Image
import time
import argparse

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers


# enable eager execution and shut up tf logging
tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)
# print("GPU available: {}".format(tf.test.is_gpu_available()))


def load_img(path):
    max_dim = 512
    img = Image.open(path)
    long_dim = max(img.size)
    scale = max_dim / long_dim
    
    # resize the image so that long side is 512 pixels
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
    img = kp_image.img_to_array(img)

    # add a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def imshow(img, title=None):
    # get rid of batch dim
    out = np.squeeze(img, axis=0)
    out = out.astype(np.uint8)
    if title:
        plt.title(title)
    plt.imshow(out)


def load_and_process_img(path):
    img = load_img(path)

    # VGG 19 uses images normalized by mean = [103.939, 116.779, 123.68] and BGR channel order
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def deprocess_img(img):
    x = img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, axis=0)
    assert len(x.shape) == 3, ("Input to deprocess_img must be an image of dimension [1, height, width, channel] (or without the '1')")

    # add means to each channel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # reverse order of channels
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


# latent content representation layers
content_layers = ['block5_conv2']

# latent style representation layers
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1',
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def get_model():
    """Creates a Keras model that takes image as input and returns needed content and style layers.
    Guys on /r/MachineLearning say that VGG gives the best results, but there is room for experimentation
    with other large models"""

    # load VGG19 pretrained on ImageNet. We do not modify the weights of the network, hence trainable=False
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # get relevant layers
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    model_outputs = content_outputs + style_outputs

    # Keras Functional API
    return models.Model(vgg.input, model_outputs)


def get_content_loss(input_features, content_features):
    """MSE loss between content representations of input image and content image"""
    return tf.reduce_mean(tf.square(input_features - content_features))


def gram_matrix(tensor):
    """Used in style loss, returns Gram matrix for given input tensor which is the inner product of vectorized feature maps"""

    # reshape the tensor to flatten all dimensions except channels
    a = tf.reshape(tensor, [-1, tensor.shape[-1]])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)   # a.T * a
    return gram / tf.cast(n, tf.float32)


def get_style_loss(input_features, gram_style_features):
    """Unlike content loss, in style loss we want to optimize not for the presence of specific features,
    but for the correlation between them, and Gram matrix is a way to do it"""
    
    h, w, c = input_features.get_shape().as_list()
    gram_input = gram_matrix(input_features)
    return tf.reduce_mean(tf.square(gram_style_features - gram_input)) # / (4. * (c**2) * (w*h)**2)


def get_tv_loss(img):
    """Total variation loss for smoothing pixel artifacts"""
    x_deltas = img[:, :, 1:, :] - img[:, :, :-1, :]
    y_deltas = img[:, 1:, :, :] - img[:, :-1, :, :]
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)


def get_feature_repr(model, content_path, style_path):
    """Get content and style representations from image paths"""

    # load the images
    content_img = load_and_process_img(content_path)
    style_img = load_and_process_img(style_path)

    # compute features for content and style images
    content_outputs = model(content_img)
    style_outputs = model(style_img)

    # extract content and style representations from model outputs
    content_features = [content_layer[0] for content_layer in content_outputs[:num_content_layers]]
    style_features = [style_layer[0] for style_layer in style_outputs[num_content_layers:]]
    return content_features, style_features


def compute_loss(model, loss_weights, init_image, content_features, gram_style_features):
    """Computes the total loss"""
    content_weight, style_weight, tv_weight = loss_weights

    # compute model outputs for init_image and extract content and style output features
    model_outputs = model(init_image)
    content_output_features = model_outputs[:num_content_layers]
    style_output_features = model_outputs[num_content_layers:]

    content_loss = 0
    style_loss = 0
    tv_loss = 0

    # add up content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_feat, out_feat in zip(content_features, content_output_features):
        content_loss += weight_per_content_layer * get_content_loss(out_feat[0], target_feat)
    
    # add up style losses from all layers
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_feat, out_feat in zip(gram_style_features, style_output_features):
        style_loss += weight_per_style_layer * get_style_loss(out_feat[0], target_feat)
    
    tv_loss += get_tv_loss(init_image)
    
    loss = (content_weight * content_loss) + (style_weight * style_loss) + (tv_weight * tv_loss)
    return loss, content_loss, style_loss, tv_loss


def compute_grads(cfg):
    """Computes gradients wrt init_image, uses a config"""
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    
    # compute grad wrt input image
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


def run_style_transfer(content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2, tv_weight=1e-4, lr=5.0, verbose=False, print_every=100):
    """The optimization loop, returns final image"""
    
    # we do not train the model
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
    if verbose: print("loaded the model")
    
    # get content and style feature representations and compute Gram matrices of style features
    content_features, style_features = get_feature_repr(model, content_path, style_path)
    gram_style_features = [gram_matrix(sf) for sf in style_features]
    if verbose: print("extracted content and style features")

    # set initial image
    init_image = load_and_process_img(content_path)
    # init_image += np.random.randn(*init_image.shape)*0.1
    init_image = tfe.Variable(init_image, dtype=tf.float32)

    # using Adam optimizer
    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.99, epsilon=1e-1)

    # config to be used
    loss_weights = (content_weight, style_weight, tv_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'content_features': content_features,
        'gram_style_features': gram_style_features
    }

    # for clipping the image after applying gradients
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    iteration = 0
    best_loss, best_img = float('inf'), None

    global_time = time.time()

    for i in range(num_iterations):
        start_time = time.time()

        # compute the gradients and perturb input image
        grads, all_loss = compute_grads(cfg)
        loss, content_loss, style_loss, tv_loss = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)

        # keep track of best result so far
        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())
        
        if verbose and (i % print_every == 0 or print_every == 1):
            print("[iteration {:4}] total loss: {:.3e}, content loss: {:.3e}, style loss: {:.3e}, tv loss: {:.3e}, time: {:.4f} s".format(
                i, loss, content_loss, style_loss, tv_loss, time.time() - start_time
            ))
        
    if verbose: print("total time: {:.4f} s".format(time.time() - global_time))
    return best_img, best_loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('-c', '--content', dest='content_path', help="path to the content image", required=True)
    parser.add_argument('-s', '--style', dest='style_path', help="path to the style image", required=True)
    parser.add_argument('-r', '--result', dest='result_path', help="path to save result image", required=True)
    parser.add_argument('-n', '--num_iter', type=int, dest='num_iter', default=1000, help="number of iterations")
    parser.add_argument('-l', '--learning_rate', type=float, dest='lr', default=5.0, help="initial learning rate to use in adam")
    parser.add_argument('--content_weight', type=float, dest='cw', default=1e4, help="content loss weight")
    parser.add_argument('--style_weight', type=float, dest='sw', default=1e-2, help="style loss weight")
    parser.add_argument('--tv_weight', type=float, dest='tw', default=1e2, help="total variation loss weight")
    parser.add_argument('-p', '--print_every', type=int, dest='print_every', default=100, help="iterations between prints")
    parser.add_argument('-v', '--verbose', dest='verbose', help="display loss every 100 iterations", action='store_true')

    args = parser.parse_args()
    result, loss = run_style_transfer(args.content_path, args.style_path, args.num_iter, args.cw, args.sw, args.tw, args.lr, args.verbose, args.print_every)
    Image.fromarray(result).save(args.result_path)
    if args.verbose: print("saved result as {}".format(args.result_path))