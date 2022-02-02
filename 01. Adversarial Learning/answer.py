
import tensorflow as tf

### Task 1: implement this function: create_adversarial_pattern()
def create_adversarial_pattern(input_image, input_label, pretrained_model, loss_object):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    ## --------- Your code ------------------
    # Get loss of the predictions.

  # Get the gradients of the loss w.r.t to the input image.

  # Get the sign of the gradients to create the perturbation

  ## --------- End. ------------------

  return signed_grad


### Task 2: create the adversarial example adv_x by 𝑎𝑑𝑣_𝑥=𝑥+𝜖∗sign(∇𝑥𝐽(𝜃,𝑥,𝑦)).
# Hint: cut the values of the model input.
def create_adv_sample(image, eps, perturbations):
  ## --------- Your code ------------------


  ## --------- End. ------------------
  return adv_x