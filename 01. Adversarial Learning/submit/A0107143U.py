
import tensorflow as tf

### Task 1: implement this function: create_adversarial_pattern()
def create_adversarial_pattern(input_image, input_label, pretrained_model, loss_object):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        ## --------- Your code ------------------
        # Get loss of the predictions.
        loss = loss_object(input_label,prediction)
    
    # Get the gradients of the loss w.r.t to the input image.
    grad = tape.gradient(loss, input_image)

    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.math.sign(grad)

    ## --------- End. ------------------
    return signed_grad


### Task 2: create the adversarial example adv_x by ððð£_ð¥=ð¥+ðâsign(âð¥ð½(ð,ð¥,ð¦)).
# Hint: cut the values of the model input.
def create_adv_sample(image, eps, perturbations):
    adv_x = image + eps*perturbations
    return adv_x