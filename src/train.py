from mobilenetV3 import MobileNetV3Small
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import losses, optimizers, utils, metrics
import tensorflow as tf

BATCH_SIZE = 64
IMAGE_SIZE = (224, 224)
EPOCHS = 20
NUM_CLASSES = 10

model = MobileNetV3Small(n_classes=NUM_CLASSES).biuld_model()
model.summary()
print(tf.config.get_visible_devices())
optimizer = optimizers.Adam()
loss_fn = losses.CategoricalCrossentropy(from_logits=True)

train_acc_metric = metrics.CategoricalAccuracy()
val_acc_metric = metrics.CategoricalAccuracy()


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = utils.to_categorical(y_train, NUM_CLASSES)
y_test = utils.to_categorical(y_test, NUM_CLASSES)

train_part = int(len(x_train)*0.8)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train[:train_part], y_train[:train_part]))
val_dataset = tf.data.Dataset.from_tensor_slices((x_train[train_part:], y_train[train_part:]))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))


def preprocess(img, y):
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 127.5 - 1.
    return img, y


train_dataset = train_dataset.map(lambda img, y: preprocess(img, y))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

val_dataset = val_dataset.map(lambda img, y: preprocess(img, y))
val_dataset = val_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

test_dataset = test_dataset.map(lambda img, y: preprocess(img, y))
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value


@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)
    loss_value = loss_fn(y, val_logits)
    return loss_value


num_batches = len(train_dataset)
best_acc = 0.0
last_loss = 0.0
not_better = 0
for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch,))
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)

        if step % 10 == 0:
            acc = train_acc_metric.result()
            print(f"Step: {step}/{num_batches} Training batch loss: {round(float(loss_value), 4)} Accuracy: {round(float(acc), 4)}")

    train_acc_metric.reset_states()

    for x_batch_val, y_batch_val in val_dataset:
        loss_value = test_step(x_batch_val, y_batch_val)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print(f'Validation accuracy at {epoch} epoch: {round(float(val_acc), 4)}')

    # custom EarlyStopping
    if epoch == 0:
        last_loss = loss_value
    elif last_loss - loss_value < 0.01:
        not_better += 1
    else:
        last_loss = loss_value

    if not_better > 5:
        break
    # save only best
    if val_acc > best_acc:
        best_acc = val_acc
        model.save_weights('src/model/')

model.load_weights('src/model/')
for x_batch_test, y_batch_test in test_dataset:
    _ = test_step(x_batch_test, y_batch_test)
test_acc = val_acc_metric.result()
val_acc_metric.reset_states()

print(f'Test accuracy: {round(float(test_acc), 4)}')
