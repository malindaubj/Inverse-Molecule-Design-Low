#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install bayesian-optimization tensorflow


# In[11]:


from bayes_opt import BayesianOptimization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l1, l2

def objective(num_layers, neurons, activation_index, reg_index, learning_rate, epochs, optimizer_index):
    
    # Map indices to actual values
    activations = ['relu', 'sigmoid', 'tanh']
    regularization = [None, l1(0.01), l2(0.01)]
    optimizers = ['adam', 'sgd']
    
    activation = activations[int(activation_index)]
    reg = regularization[int(reg_index)]
    optimizer = optimizers[int(optimizer_index)]
    
    input_vec = Input(shape=(50, 35))
    x = Flatten()(input_vec)
    
    for _ in range(int(num_layers)):
        x = Dense(int(neurons), activation=activation, kernel_regularizer=reg)(x)
        if reg_index == 0:
            pass  # No regularization
        elif reg_index == 1:
            x = Dropout(0.5)(x)  # Dropout regularization
    
    encoded = Dense(6*6*2, activation=activation)(x)
    
    # Decoder (Mirroring the encoder structure)
    x = encoded
    for _ in range(int(num_layers)):
        x = Dense(int(neurons), activation=activation, kernel_regularizer=reg)(x)
        if reg_index == 0:
            pass  # No regularization
        elif reg_index == 1:
            x = Dropout(0.5)(x)  # Dropout regularization

    x = Dense(50*35, activation='sigmoid')(x)
    decoded = Reshape((50, 35))(x)
    
    autoencoder = Model(input_vec, decoded)
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
    
    history = autoencoder.fit(data, data, epochs=int(epochs), batch_size=64, validation_split=0.1, verbose=0)
    
    # Return negative validation loss for maximization
    return -history.history['val_loss'][-1]


# In[12]:


pbounds = {
    'num_layers': (15, 40),  # example range, you can adjust based on your needs
    'neurons': (64, 512),
    'activation_index': (0, 2),
    'reg_index': (0, 2),
    'learning_rate': (1e-6, 1e-1),
    'epochs': (500, 1200),
    'optimizer_index': (0, 1)
}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=1,
)


# In[13]:


import pickle
with open(r'D:\Studies\PHD\Research\Study\Study 4.1\Project\DataProcessing\CanonicalSmiles1.pickle', 'rb') as f:
       X, SMILES, Y = pickle.load(f)
        
data = X


# In[14]:


import numpy as np
import random
selected_indices = random.sample(range(X.shape[0]), 10)
selected_X = X[selected_indices]
selected_X


# In[19]:


from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
import numpy as np
from tqdm import tqdm
import pandas as pd
# Create dummy binary data
#data = np.random.randint(2, size=(1000, 50, 35))
data = selected_X

optimizer.maximize(
    init_points=5,
    n_iter=10,
)      

#with open("results.txt", "w") as file:
#    for result in optimizer.res:
#        file.write(str(result) + "\n")

# Convert the results list of dictionaries to a DataFrame
df = pd.DataFrame(optimizer.res)

# Save the DataFrame to a CSV file
df.to_csv("results.csv", index=False)


# In[23]:


print(optimizer.max)
with open("best_result.txt", "w") as file:
    for key, value in optimizer.max.items():
        file.write(f"{key}: {value}\n")


# In[21]:


import matplotlib.pyplot as plt

# Extract values from the optimization results
activation_indices = [x['params']['activation_index'] for x in optimizer.res]
epochs_values = [x['params']['epochs'] for x in optimizer.res]
learning_rates = [x['params']['learning_rate'] for x in optimizer.res]
neurons_values = [x['params']['neurons'] for x in optimizer.res]
num_layers_values = [x['params']['num_layers'] for x in optimizer.res]
optimizer_indices = [x['params']['optimizer_index'] for x in optimizer.res]
reg_indices = [x['params']['reg_index'] for x in optimizer.res]
target_values = [x['target'] for x in optimizer.res]

# Plot hyperparameters
fig, axs = plt.subplots(7, 1, figsize=(15, 20))

axs[0].plot(activation_indices, '-o')
axs[0].set_title('Activation Index over Iterations')
axs[0].set_ylabel('Activation Index')

axs[1].plot(epochs_values, '-o')
axs[1].set_title('Epochs over Iterations')
axs[1].set_ylabel('Epochs')

axs[2].plot(learning_rates, '-o')
axs[2].set_title('Learning Rate over Iterations')
axs[2].set_ylabel('Learning Rate')

axs[3].plot(neurons_values, '-o')
axs[3].set_title('Neurons per Layer over Iterations')
axs[3].set_ylabel('Neurons per Layer')

axs[4].plot(num_layers_values, '-o')
axs[4].set_title('Number of Layers over Iterations')
axs[4].set_ylabel('Number of Layers')

axs[5].plot(optimizer_indices, '-o')
axs[5].set_title('Optimizer Index over Iterations')
axs[5].set_ylabel('Optimizer Index')

axs[6].plot(reg_indices, '-o')
axs[6].set_title('Regularization Index over Iterations')
axs[6].set_ylabel('Regularization Index')

plt.tight_layout()

# Save the plot as a PNG
plt.savefig('hyperparameters_plot.png', dpi=300, bbox_inches='tight')

#plt.show()


# In[22]:





# In[ ]:


# # {'target': -0.6937342286109924, 'params': {'activation_index': 2.0, 'epochs': 500.0, 'learning_rate': 1e-06, 
# #                                            'neurons': 393.9534025942016, 'num_layers': 1.0, 'optimizer_index': 0.0,
# #                                            'reg_index': 2.0}}

# # {'target': -0.6937878131866455, 'params': {'activation_index': 1.5303725262384222, 'epochs': 554.7587623519765, 
# #                                            'learning_rate': 0.05219851882346603, 'neurons': 408.17162203905775, 
# #                                            'num_layers': 20.927036024553406, 'optimizer_index': 0.13298154074487634, 
# #                                            'reg_index': 0.40384924203586325}}

# # {'target': -0.06987646967172623, 'params': {'activation_index': 1.4051557022918577, 'epochs': 618.7202258885221, 
# #                                             'learning_rate': 0.0026442544585613898, 'neurons': 108.25560715344866, 
# #                                             'num_layers': 27.715492223331783, 'optimizer_index': 0.7703341947670929, 
# #                                             'reg_index': 0.4830045265645564}}

# {'target': -0.06389316916465759, 'params': {'activation_index': 1.8501976935442401, 'epochs': 919.0962665675806,
#                                             'learning_rate': 0.0007594691827881292, 'neurons': 464.6111106194393, 
#                                             'num_layers': 1.155621851574995, 'optimizer_index': 0.5087434877670002, 
#                                             'reg_index': 0.8889961232166153}}


# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.regularizers import l1, l2

# # Hyperparameters from Bayesian Optimization
# activation_function = 'tanh'
# epochs = 619
# learning_rate = 0.0026
# neurons_per_layer = 108
# num_layers = 27
# optimizer = 'adam'
# regularization = l2(0.4)

# # Autoencoder architecture
# input_vec = Input(shape=(50, 35))
# x = Flatten()(input_vec)

# # Encoding layers
# for _ in range(num_layers):
#     x = Dense(neurons_per_layer, activation=activation_function, kernel_regularizer=regularization)(x)

# # Latent representation
# latent_dim = 6 * 6 * 2
# x = Dense(latent_dim, activation=activation_function)(x)

# # Decoding layers
# for _ in range(num_layers):
#     x = Dense(neurons_per_layer, activation=activation_function, kernel_regularizer=regularization)(x)

# # Final decoding to original dimensions
# decoded = Dense(50*35, activation='sigmoid')(x)
# decoded = Reshape((50, 35))(decoded)

# # Model compilation
# autoencoder = Model(input_vec, decoded)
# autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')

# # Model summary to check the architecture
# autoencoder.summary()


# In[ ]:


# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.regularizers import l2

# # Create a synthetic binary dataset
# #data = np.random.randint(2, size=(1000, 50, 35))

# # Hyperparameters from Bayesian Optimization
# # activation_function = 'tanh'
# # epochs = 500
# # learning_rate = 1e-06
# # neurons_per_layer = 394
# # num_layers = 1
# # optimizer = 'adam'
# # regularization = l2(0.01)

# # Autoencoder architecture
# input_vec = Input(shape=(50, 35))
# x = Flatten()(input_vec)

# # Encoding layers
# for _ in range(num_layers):
#     x = Dense(neurons_per_layer, activation=activation_function, kernel_regularizer=regularization)(x)

# # Latent representation
# latent_dim = 6 * 6 * 2
# x = Dense(latent_dim, activation=activation_function)(x)

# # Decoding layers
# for _ in range(num_layers):
#     x = Dense(neurons_per_layer, activation=activation_function, kernel_regularizer=regularization)(x)

# # Final decoding to original dimensions
# decoded = Dense(50*35, activation='sigmoid')(x)
# decoded = Reshape((50, 35))(decoded)

# # Model compilation
# autoencoder = Model(input_vec, decoded)
# autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')

# # Train the autoencoder
# autoencoder.fit(data, data, epochs=epochs, batch_size=64, validation_split=0.1, verbose=1)

# # Predict outputs on the dataset
# decoded_data = autoencoder.predict(data)

# # To get binary values (0 or 1), you can threshold the predictions
# threshold = 0.5
# binary_decoded_data = (decoded_data > threshold).astype(int)

# # Print the first original and decoded sample to see the results
# print("Original Data:\n", data[0])
# print("\nDecoded Data:\n", binary_decoded_data[0])


# In[ ]:


# # Calculate the accuracy
# matching_elements = np.sum(data == binary_decoded_data)
# total_elements = data.size
# accuracy_percentage = (matching_elements / total_elements) * 100

# print(f"Accuracy: {accuracy_percentage:.2f}%")


# In[ ]:




