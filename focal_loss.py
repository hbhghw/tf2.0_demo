import tensorflow as tf

def focal_loss(y_true,y_pred,alpha=0.25,gamma=2):
    y_pred = tf.clip_by_value(y_pred, 1e-8, 1 - 1e-8)
    loss_p = (1-y_pred)**gamma*y_true*tf.math.log(y_pred)
    loss_n = y_pred**gamma*(1-y_true)*tf.math.log(1-y_pred)
    return -tf.reduce_mean(alpha*loss_p + (1-alpha)*loss_n)
