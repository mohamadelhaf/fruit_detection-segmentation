#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importer les bibliothèques nécessaires
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


# In[3]:


# Définir les fonctions pour calculer Dice Coefficient et IoU
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calcule le Dice Coefficient entre les masques réels (y_true) et les masques prédits (y_pred).
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou(y_true, y_pred, smooth=1e-6):
    """
    Calcule l'IoU (Intersection over Union) entre les masques réels (y_true) et les masques prédits (y_pred).
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Chemin du dataset
dataset_path = r"C:\Users\etudiant\Downloads\archive (3)\FruitSegmentationDataset\FruitSeg30"

# Fonction pour charger les images et les masques
def load_images_and_masks(dataset_path, img_size=(256, 256)):
    images = []
    masks = []
    
    # Parcourir chaque dossier de fruit
    for fruit_folder in os.listdir(dataset_path):
        fruit_path = os.path.join(dataset_path, fruit_folder)
        # Vérifier si c'est un dossier
        if not os.path.isdir(fruit_path):
            print(f"Attention : {fruit_folder} n'est pas un dossier")
            continue
        
        # Chemins des dossiers Images et Mask
        img_folder = os.path.join(fruit_path, "Images")
        mask_folder = os.path.join(fruit_path, "Mask")
        
        # Vérifier si les dossiers Images et Mask existent
        if not os.path.exists(img_folder):
            print(f"Attention : Dossier Images manquant pour {fruit_folder}")
            continue
        if not os.path.exists(mask_folder):
            print(f"Attention : Dossier Mask manquant pour {fruit_folder}")
            continue
        
        # Parcourir les images dans le dossier Images
        for img_name in os.listdir(img_folder):
            img_path = os.path.join(img_folder, img_name)
            
            # Extraire le nom de base (sans extension)
            base_name = os.path.splitext(img_name)[0]  # Retire l'extension (.png, .jpg, etc.)
            
            # Construire le nom du fichier de masque
            mask_name = f"{base_name}_mask.png"  # Ajoute le suffixe _mask et l'extension .png
            mask_path = os.path.join(mask_folder, mask_name)
            
            # Afficher les chemins pour le débogage
            print(f"Image : {img_path}")
            print(f"Masque attendu : {mask_path}")
            
            # Vérifier si le fichier de masque existe
            if not os.path.exists(mask_path):
                print(f"Attention : Masque manquant pour {img_name} dans {fruit_folder}")
                continue
            
            # Charger l'image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Erreur : Impossible de charger l'image {img_path}")
                continue
            
            # Charger le masque
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Erreur : Impossible de charger le masque {mask_path}")
                continue
            
            # Redimensionner et normaliser
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalisation
            
            mask = cv2.resize(mask, img_size)
            mask = mask / 255.0  # Normalisation
            mask = np.expand_dims(mask, axis=-1)  # Ajouter une dimension pour la segmentation
            
            images.append(img)
            masks.append(mask)
    
    if len(images) == 0:
        print("Aucune image ou masque n'a été chargé. Vérifiez les chemins d'accès et les formats de fichiers.")
    
    return np.array(images), np.array(masks)

# Charger les données
images, masks = load_images_and_masks(dataset_path)

# Vérifier si des données ont été chargées
if len(images) > 0:
    # Diviser en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    print(f"Nombre d'images d'entraînement : {X_train.shape[0]}")
    print(f"Nombre d'images de validation : {X_val.shape[0]}")
else:
    print("Aucune donnée à diviser. Veuillez vérifier les messages d'erreur ci-dessus.")


# In[4]:


# Construction du modèle U-Net
def unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # Middle
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    # Decoder
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(256, 2, activation='relu', padding='same')(up5)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(128, 2, activation='relu', padding='same')(up6)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(64, 2, activation='relu', padding='same')(up7)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # Output
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Créer le modèle
model = unet()

# Compiler le modèle avec les métriques personnalisées
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', dice_coefficient, iou])

# Afficher un résumé du modèle
model.summary()


# In[6]:


# Callback pour sauvegarder le meilleur modèle
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Entraînement du modèle
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=[checkpoint])

# Récupérer l'historique des métriques
epochs = history.epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
# Supposons que l'IoU et le Dice sont définis dans les métriques
iou = history.history['iou'] if 'iou' in history.history else [np.nan] * len(epochs)  # Si l'IoU n'est pas dans les métriques, on remplit avec NaN
dice = history.history['dice'] if 'dice' in history.history else [np.nan] * len(epochs)  # Même chose pour le Dice

# Créer la figure pour afficher les courbes
plt.figure(figsize=(10, 8))

# IoU
plt.plot(epochs, iou, label='IoU', color='blue')
# Loss
plt.plot(epochs, loss, label='Loss', color='red')
# Accuracy
plt.plot(epochs, accuracy, label='Accuracy', color='green')
# Dice
plt.plot(epochs, dice, label='Dice', color='purple')

# Ajouter des étiquettes et un titre
plt.xlabel('Epochs')
plt.ylabel('Valeurs')
plt.title('IoU, Loss, Accuracy, Dice en fonction des epochs')

# Ajouter une légende
plt.legend()

# Afficher le graphique
plt.show()
# Récupérer l'historique des métriques
epochs = history.epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
# Supposons que l'IoU et le Dice sont définis dans les métriques
iou = history.history['iou'] if 'iou' in history.history else [np.nan] * len(epochs)  # Si l'IoU n'est pas dans les métriques, on remplit avec NaN
dice = history.history['dice'] if 'dice' in history.history else [np.nan] * len(epochs)  # Même chose pour le Dice



# In[10]:


import matplotlib.pyplot as plt

# Données d'exemple tirées des logs d'entraînement
epochs = list(range(1, 38))  # Nombre d'époques (ici 50 époques)

# Remplis ces listes avec les valeurs correspondantes extraites de tes logs
accuracy = [0.6864, 0.6840, 0.6920, 0.6893, 0.6951, 0.7014, 0.7049, 0.7026, 0.7106, 0.7123, 0.7078, 0.7085, 0.7092, 0.7114, 0.7189, 0.7225, 0.7258, 0.7286, 0.7329, 0.7352, 0.7458, 0.7574, 0.7628, 0.7727, 0.7785, 0.7801, 0.7825, 0.7901, 0.7979, 0.8000, 0.8005, 0.8008, 0.8011, 0.8019, 0.8069, 0.8090, 0.8109]
dice_coefficient = [0.8210, 0.8634, 0.8459, 0.8797, 0.8874, 0.8946, 0.9154, 0.9337, 0.9370, 0.9401, 0.9374, 0.9393, 0.9504, 0.9487, 0.9566, 0.9338, 0.9554, 0.9579, 0.9641, 0.9658, 0.9611, 0.9651, 0.9667, 0.9665, 0.9652, 0.9282, 0.9558, 0.9621, 0.9697, 0.9692, 0.9686, 0.9738, 0.9726, 0.9709, 0.9749, 0.9734, 0.9735]
iou = [0.7030, 0.7611, 0.7361, 0.7866, 0.7989, 0.8105, 0.8449, 0.8762, 0.8820, 0.8873, 0.8830, 0.8862, 0.9057, 0.9031, 0.9170, 0.8766, 0.9150, 0.9195, 0.9310, 0.9341, 0.9255, 0.9327, 0.9358, 0.9353, 0.9331, 0.8669, 0.9156, 0.9272, 0.9413, 0.9403, 0.9393, 0.9490, 0.9468, 0.9435, 0.9510, 0.9483, 0.9544]
loss = [0.1985, 0.1634, 0.1811, 0.1449, 0.1347, 0.1243, 0.0996, 0.0814, 0.0771, 0.0728, 0.0762, 0.0718, 0.0605, 0.0633, 0.0545, 0.0797, 0.0535, 0.0527, 0.0448, 0.0431, 0.0451, 0.0422, 0.0396, 0.0412, 0.0438, 0.0940, 0.0533, 0.0460, 0.0383, 0.0378, 0.0389, 0.0334, 0.0342, 0.0367, 0.0321, 0.0338, 0.0301]


# Tracer les courbes
plt.figure(figsize=(10, 8))

# IoU
plt.plot(epochs, iou, label='IoU', color='blue')

# Loss
plt.plot(epochs, loss, label='Loss', color='red')

# Accuracy
plt.plot(epochs, accuracy, label='Accuracy', color='green')

# Dice
plt.plot(epochs, dice_coefficient, label='Dice', color='purple')

# Ajouter des étiquettes et un titre
plt.xlabel('Epochs')
plt.ylabel('Valeurs')
plt.title('IoU, Loss, Accuracy, Dice en fonction des epochs')

# Ajouter une légende
plt.legend()

# Afficher le graphique
plt.show()



# In[5]:


print(len(epochs))
print(len(iou))
print(len(loss))
print(len(accuracy))
print(len(dice_coefficient))


# In[7]:


# Charger le meilleur modèle
model = tf.keras.models.load_model('best_model.h5', custom_objects={'dice_coefficient': dice_coefficient, 'iou': iou})

# Prédire sur l'ensemble de validation
y_pred = model.predict(X_val)

# Afficher quelques résultats
def plot_results(images, masks, preds, num_samples=5):
    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 3, 3*i + 1)
        plt.imshow(images[i])
        plt.title("Image originale")
        plt.axis('off')

        plt.subplot(num_samples, 3, 3*i + 2)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title("Masque réel")
        plt.axis('off')

        plt.subplot(num_samples, 3, 3*i + 3)
        plt.imshow(preds[i].squeeze(), cmap='gray')
        plt.title("Masque prédit")
        plt.axis('off')
    plt.show()

plot_results(X_val, y_val, y_pred)


# In[8]:


# Sauvegarder le modèle
model.save('fruit_segmentation_unet.h5')


# In[9]:


# Charger et utiliser le modèle
loaded_model = tf.keras.models.load_model('fruit_segmentation_unet.h5', custom_objects={'dice_coefficient': dice_coefficient, 'iou': iou})
predictions = loaded_model.predict(X_val)


# In[ ]: