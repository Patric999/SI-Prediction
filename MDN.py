# Add required imports
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from copy import deepcopy

def create_book_example(n=1000):
    """bementei szinuiszborzalom generálása
    ezt majd felváltja az igazi adatok beolvasása
    ez így a #1 TODO"""
    # sample uniformly over the interval (0,1)
    X = np.random.uniform(0., 1., (n,1)).astype(np.float32)
    # target values
    y = X + 0.3 * np.sin(2 * np.pi * X) + np.random.uniform(-0.1, 0.1, size=(n,1)).astype(np.float32)
    # test data
    x_test = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)
    return X, y, x_test

"""bemenet generálása és kirajzolása"""
X, y, x_test = create_book_example(n=4000)
flipped_x = deepcopy(y)
flipped_y = deepcopy(X)
plt.plot(flipped_x, flipped_y, 'ro', alpha=0.04)
plt.show() # bemeneti sinus kirajzolasa

print(X.shape, y.shape)


"""Itt hozzuk létre a MDN networkot"""
# In our toy example, we have single input feature
l = 1
# Number of gaussians to represent the multimodal distribution
k = 26

# Mixture Density Network
input = tf.keras.Input(shape=(l,))
layer = tf.keras.layers.Dense(50, activation='tanh', name='baselayer')(input)
mu = tf.keras.layers.Dense((l * k), activation=None, name='mean_layer')(layer)
# variance (should be greater than 0 so we exponentiate it)
var_layer = tf.keras.layers.Dense(k, activation=None, name='dense_var_layer')(layer)
var = tf.keras.layers.Lambda(lambda x: tf.math.exp(x), output_shape=(k,), name='variance_layer')(var_layer)
# mixing coefficient should sum to 1.0
pi = tf.keras.layers.Dense(k, activation='softmax', name='pi_layer')(layer)

model = tf.keras.models.Model(input, [pi, mu, var])
optimizer = tf.keras.optimizers.Adam()
model.summary()

def calc_pdf(y, mu, var):
    """Calculate component density
    matekozás
    a kimeneti gauszok kiértékelése"""
    value = tf.subtract(y, mu)**2
    value = (1/tf.math.sqrt(2 * np.pi * var)) * tf.math.exp((-1/(2*var)) * value)
    return value

def mdn_loss(y_true, pi, mu, var):
    """MDN Loss Function
    Ezt majd olyanra kell megírni, hogy SI predikcio kompatibilis legyen
    lehet egy négyzetes legkisebb eltéréses függvény is jó
    Ez alapján fog optimalizálni a network a betanításkor
    """
    out = calc_pdf(y_true, mu, var)
    # multiply with each pi and sum it
    out = tf.multiply(out, pi)
    out = tf.reduce_sum(out, 1, keepdims=True)
    out = -tf.math.log(out + 1e-10)
    return tf.reduce_mean(out)

# calc_pdf(3.0, 0.0, 1.0).numpy()
calc_pdf(np.array([3.0]), np.array([0.0, 0.1, 0.2]), np.array([1.0, 2.2, 3.3])).numpy()

# Numpy version
def pdf_np(y, mu, var):
    n = np.exp((-(y-mu)**2)/(2*var))
    d = np.sqrt(2 * np.pi * var)
    return n/d
"""Ellenőrzés, nem igazán vágom hogy pont itt mi van"""
print('Numpy version: ')
pdf_np(3.0, np.array([0.0, 0.1, 0.2]), np.array([1.0, 2.2, 3.3]))

loss_value = mdn_loss(
    np.array([3.0, 1.1]).reshape(2,-1).astype('float64'),
    np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).reshape(2,-1).astype('float64'),
    np.array([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]).reshape(2,-1).astype('float64'),
    np.array([[1.0, 2.2, 3.3], [1.0, 2.2, 3.3]]).reshape(2,-1).astype('float64')
).numpy()
assert np.isclose(loss_value, 3.4714, atol=1e-5), 'MDN loss incorrect'

"""
itt kezdődik el a training
ha jól sejtem itt csak megetetjük az adatokat vele
és a későbbiekben sem kell sokat majd módosítani"""

# Use Dataset API to load numpy data (load, shuffle, set batch size)
# adatok betöltése
N = flipped_x.shape[0]
dataset = tf.data.Dataset \
    .from_tensor_slices((flipped_x, flipped_y)) \
    .shuffle(N).batch(N)


def train_step(model, optimizer, train_x, train_y):
    """tanítófüggvény lépése
    beadunk adatokat,
    loss-t számolunk
    a loss mértéke alapján módosításokat végzünk a belső változókon
    visszaadjuk a loss-t"""
    # GradientTape: Trace operations to compute gradients
    with tf.GradientTape() as tape:
        pi_, mu_, var_ = model(train_x, training=True)
        # calculate loss
        loss = mdn_loss(train_y, pi_, mu_, var_)
    # compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

losses = []
EPOCHS = 1500 #jó sokat tanítjuk, eredetileg 6000 volt, most hogy jobban fusson levettem 1500-ra
"""epochs változót majd ki kéne tenni az elejére, hogy egy helyen legyenek módosíthatóak a fontos változók"""
print_every = int(0.1 * EPOCHS)

# Define model and optimizer
model = tf.keras.models.Model(input, [pi, mu, var])
optimizer = tf.keras.optimizers.Adam()

# Start training
"""for ciklussal végigmegyünk az epoch-okon 
használjuk a megírt loss és train_step függvényeinket"""
print('Print every {} epochs'.format(print_every))
for i in range(EPOCHS):
    for train_x, train_y in dataset:
        loss = train_step(model, optimizer, train_x, train_y)
        losses.append(loss)
    if i % print_every == 0:
        print('Epoch {}/{}: loss {}'.format(i, EPOCHS, losses[-1]))

"""kiértékelés... 
a loss miért tart -1-hez?
eredetiben is így volt, de na"""
plt.plot(range(len(losses)), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training loss')
plt.show()

"""Most jönnek a predikciók
ehhez a részhez gondolom azért részben hozzá kéne nyúlni,
ha máshoz nem, legalább a kirajzoláshoz, 
hogy van-e értelme a plottolásnak SI predikció esetén is"""
def approx_conditional_mode(pi, var, mu):
    """Approx conditional mode
    Because the conditional mode for MDN does not have simple analytical
    solution, an alternative is to take mean of most probable component
    at each value of x (PRML, page 277)
    """
    n, k = pi.shape
    out = np.zeros((n, l))
    # Get the index of max pi value for each row
    max_component = tf.argmax(pi, axis=1)
    for i in range(n):
        # The mean value for this index will be used
        mc = max_component[i].numpy()
        for j in range(l):
            out[i, j] = mu[i, mc*(l+j)]
    return out


# Get predictions
pi_vals, mu_vals, var_vals = model.predict(x_test)
pi_vals.shape, mu_vals.shape, var_vals.shape

# Get mean of max(mixing coefficient) of each row
preds = approx_conditional_mode(pi_vals, var_vals, mu_vals)

# Plot along with training data
fig = plt.figure(figsize=(8, 8))
plt.plot(flipped_x, flipped_y, 'ro')
plt.plot(x_test, preds, 'g.')
# plt.plot(flipped_x, preds2, 'b.')
plt.show()


def sample_predictions(pi_vals, mu_vals, var_vals, samples=10):
    n, k = pi_vals.shape
    # print('shape: ', n, k, l)
    # place holder to store the y value for each sample of each row
    out = np.zeros((n, samples, l))
    for i in range(n):
        for j in range(samples):
            # for each sample, use pi/probs to sample the index
            # that will be used to pick up the mu and var values
            idx = np.random.choice(range(k), p=pi_vals[i])
            for li in range(l):
                # Draw random sample from gaussian distribution
                out[i,j,li] = np.random.normal(mu_vals[i, idx*(li+l)], np.sqrt(var_vals[i, idx]))
    return out

sampled_predictions = sample_predictions(pi_vals, mu_vals, var_vals, 10)

import matplotlib.patches as mpatches

fig = plt.figure(figsize=(6, 6))
plt.plot(flipped_x, flipped_y, 'ro', label='train')
for i in range(sampled_predictions.shape[1]):
    plt.plot(x_test, sampled_predictions[:, i], 'g.', alpha=0.3, label='predicted')
patches = [
    mpatches.Patch(color='green', label='Training'),
    mpatches.Patch(color='red', label='Predicted')
]

plt.legend(handles=patches)
plt.show()    
