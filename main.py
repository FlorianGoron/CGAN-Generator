from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import ELU, PReLU, LeakyReLU
from keras.layers import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers.legacy import Adam

from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import datetime
import argparse
import json
import os 

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class CGAN():
    def __init__(self):
        # Format d'entrée
        self.workspace_path= os.getcwd()
        self.json_data_path=self.workspace_path+'/use_case/match_1.json'
        self.input_signal = 150
        self.channels = 1
        self.signal_shape = (self.input_signal, self.channels)
        self.num_classes = 8
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Construire et compiler le discriminateur
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Construire le générateur
        self.generator = self.build_generator()

        # Le générateur prend en entrée le bruit et l'étiquette cible
        # et génère le chiffre correspondant à cette étiquette
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        signal = self.generator([noise, label])

        # Pour le modèle combiné, nous n'entraînerons que le générateur
        self.discriminator.trainable = False

        # Le discriminateur prend l'image générée en entrée et détermine la validité
        # et l'étiquette de cette image
        valid = self.discriminator([signal, label])

        # Le modèle combiné (générateur et discriminateur empilés)
        # Entraîne le générateur à tromper le discriminateur
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.signal_shape), activation='tanh'))
        model.add(Reshape(self.signal_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        signal = model(model_input)

        return Model([noise, label], signal)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.signal_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        signal = Input(shape=self.signal_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.signal_shape))(label))
        flat_signal = Flatten()(signal)

        model_input = multiply([flat_signal, label_embedding])

        validity = model(model_input)

        return Model([signal, label], validity)

    def data_pre_processing(self):
        # Chargement de l'ensemble de données
        with open(self.json_data_path, 'r') as f:
            data = f.read()
            json_data = json.loads(data)

        # Transformons la liste de dictonnaire imbriqué en pd.DataFrame
        df = pd.DataFrame(json_data)
        
        
        filtered_df = df.loc[(df['norm'].str.len() >= 5) & (df['norm'].str.len() <= 150)]

        self.X_train = filtered_df['norm'].values

        # On encode les labels 
        self.y_train = LabelEncoder().fit_transform(filtered_df['label'].values)

        # {'walk' : 7, 'run' : 4, 'tackle' : 6, 'dribble' : 1, 'pass' : 2, 'rest' : 3, 'cross' : 0, 'shot' : 5}

        # Afin de mieux faire comprendre à l'algo CGAN que je souhaite avoir des vecteurs de taille inférieur à 150 et que chaque action à un domaine de longueur de prédilection nous allons rajouter des 0 aux norm de moins de 150 pas de temps.
        self.X_train = np.array([lst if len(lst) == 150 else (lst + [0.0] * (150 - len(lst))) for lst in self.X_train if len(lst) <= 150])

        self.demi_max = np.max(self.X_train)/2

        return self.X_train, self.y_train

    def train(self, epochs, batch_size=128, sample_interval=50):

     
        self.X_train, self.y_train = self.data_pre_processing()
        
        # Normalisation (les données de norm sont maintenant comprisent entre [-1:1])
        self.X_train = (self.X_train.astype(np.float32) - self.demi_max) / self.demi_max
        self.y_train = self.y_train.reshape(-1, 1)

        # Vérités de terrain contradictoires
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Mesure de l'apprentissage
        self.d_loss_historised = []
        self.accuracy_historised = []
        self.g_loss_historised = []

        for epoch in range(epochs):

            # ---------------------
            #  Entraînement du discriminateur
            # ---------------------

            # Sélection d'un demi lot de signaux au hasard
            idx = np.random.randint(0, self.X_train.shape[0], batch_size)
            signals, labels = self.X_train[idx], self.y_train[idx]

            # Bruit de l'échantillon comme entrée du générateur
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Générer un demi lot de nouveaux signaux
            gen_signals = self.generator.predict([noise, labels])

            # Entraînement du discriminateur
            d_loss_real = self.discriminator.train_on_batch([signals, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_signals, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Entraînement du Générateur
            # ---------------------

            # Condition sur les labels
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)

            # Entraînement du Générateur
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Tracer la progression
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # Stockage des pertes
            self.d_loss_historised.append(d_loss[0])
            self.accuracy_historised.append(100*d_loss[1])
            self.g_loss_historised.append(g_loss)

            # A l'intervalle de sauvegarde => sauvegarde des échantillons
            if epoch % sample_interval == 0:
                self.sample_signal(epoch) 


    def sample_signal(self, epoch):
        # Tirage d'un bruit 
        noise = np.random.normal(0, 1, (self.num_classes, 100))
        sampled_labels = np.arange(0, self.num_classes).reshape(-1, 1)
        gen_signals = self.generator.predict([noise, sampled_labels])

        # Rééchelonnement 0 - max
        gen_signals = self.demi_max * gen_signals + self.demi_max

        # Recupération de la loss du générateur et du discriminateur par epoch
        learning = pd.DataFrame(data={'d_loss' : self.d_loss_historised, 'acc.' : self.accuracy_historised, 'g_loss' : self.g_loss_historised})

        # {'cross' : 0, 'dribble' : 1, 'pass' : 2, 'rest' : 3, 'run' : 4, 'shot' : 5, 'tackle' : 6, 'walk' : 7}

        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=("cross", "dribble", "D_loss", "pass",  "rest",  "Accuracy", "run", "shot", "G_loss", "tackle", "walk"))

        fig.add_trace(go.Scatter(x=list(range(1, 151)), y=gen_signals[0,:,0]),
                    row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(1, 151)), y=gen_signals[1,:,0]),
                    row=1, col=2)

        fig.add_trace(go.Scatter(x=list(range(1, 151)), y=gen_signals[2,:,0]),
                    row=2, col=1)

        fig.add_trace(go.Scatter(x=list(range(1, 151)), y=gen_signals[3,:,0]),
                    row=2, col=2)

        fig.add_trace(go.Scatter(x=list(range(1, 151)), y=gen_signals[4,:,0]),
                    row=3, col=1)

        fig.add_trace(go.Scatter(x=list(range(1, 151)), y=gen_signals[5,:,0]),
                    row=3, col=2)

        fig.add_trace(go.Scatter(x=list(range(1, 151)), y=gen_signals[6,:,0]),
                    row=4, col=1)

        fig.add_trace(go.Scatter(x=list(range(1, 151)), y=gen_signals[7,:,0]),
                    row=4, col=2)

        fig.add_trace(go.Scatter(x=list(learning.index), y=learning['d_loss']),
                    row=1, col=3)

        fig.add_trace(go.Scatter(x=list(learning.index), y=learning['acc.']),
                    row=2, col=3)
        
        fig.add_trace(go.Scatter(x=list(learning.index), y=learning['g_loss']),
                    row=3, col=3)

        fig.update_layout(height=1000, width=1400,
                        title_text="Actions créées avec CGAN")
        fig.write_html(self.workspace_path+"images/%d.html" % epoch)

    def generate_match(self, nombre_de_matchs=1, nombre_de_minutes=15, style_de_jeu='Equilibré', model_path=None):

        ### Calcul de la matrice de transition en fontion du style de jeu ###
        
        # Pour redistribuer les signaux de façon cohérente nous allons chercher la chaine de markov associé au match 1
    
        # Initialisez une matrice de transition avec des zéros
        transition_matrix = np.zeros((self.num_classes, self.num_classes))

        self.X_train, self.y_train = self.data_pre_processing()

        sequence = self.y_train.flatten()
        for i in range(len(sequence) - 1):
            current_state = sequence[i]
            next_state = sequence[i + 1]
            transition_matrix[current_state][next_state] += 1

        # Normalisons chaque ligne pour garantir que chaque ligne somme à 1
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

        alpha = 0.3  # Paramètre d'autementation

        # Créons une matrice d'ajustement
        adjustment_matrix = np.identity(8)

        if style_de_jeu == 'Offensif':
            # Nous voulions augmenter les probabilités pour les états tire et passe (indexés à 2 et 5 en Python)
            adjustment_matrix[2, 2] += alpha
            adjustment_matrix[5, 5] += alpha

        elif style_de_jeu == 'Defensif':    
            # Nous voulions augmenter les probabilités pour les états run et walk et tackle (indexés à 4, 6 et 7 en Python)
            adjustment_matrix[4, 4] += alpha
            adjustment_matrix[6, 6] += alpha
            adjustment_matrix[7, 7] += alpha

        # Multiplions la matrice de transition par la matrice d'ajustement
        transition_matrix = np.dot(transition_matrix, adjustment_matrix)

        # Normalisons chaque ligne pour garantir que chaque ligne somme à 1
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1)[:, np.newaxis]
      
        #########  Generation du signal  ######### 
           
        # En moyenne la durée d'une action dans mon jeux d'entraitement est d'environ 50 pas de temps (1s)
        nombre_de_signaux = nombre_de_minutes * 60
        
        for _ in range(nombre_de_matchs):

            # Partons du label 'walk' comme état initial 
            initial_state = 7

            # Créons une liste pour stocker les labels 
            labels = [initial_state]

            # Génération de la série temporelle des labels
            current_state = initial_state
            for _ in range(nombre_de_signaux-1):
                next_state = np.random.choice(range(len(transition_matrix)), p=transition_matrix[current_state])
                labels.append(next_state)
                current_state = next_state
      
            # Generer les signaux 
            noise = np.random.normal(0, 1, (nombre_de_signaux, 100))  
            labels = np.array(labels).reshape(-1, 1)

            # On ne charge le model enregistré que si model_path est non vide
            if model_path != None :
                self.generator = load_model(model_path)

            generated_signals = self.generator.predict([noise, labels])

            # Remettre à l'échelle
            generated_signals = self.demi_max * generated_signals + self.demi_max


            # Suppression des valeurs proches de 0.0 par la fin
            threshold = 0.001
            consecutive_count_threshold = 10

            def truncate_array(arr):
                consecutive_count = 0
                for i in range(len(arr)):
                    if abs(arr[i]) < threshold:
                        consecutive_count += 1
                        if consecutive_count == consecutive_count_threshold:
                            return arr[:i+1-consecutive_count_threshold]
                    else:
                        consecutive_count = 0
                return arr

            truncate_generated_signals = []
            for sub_array in generated_signals:
                truncate_generated_signals.append(truncate_array(sub_array))

            # Transformation en dictionnaire

            # Inverser le dictionnaire
            mapping = {'cross' : 0, 'dribble' : 1, 'pass' : 2, 'rest' : 3, 'run' : 4, 'shot' : 5, 'tackle' : 6, 'walk' : 7}
            inverse_mapping = {v: k for k, v in mapping.items()}

            # Remplacer les valeurs de l'array par les clés du dictionnaire inversé
            labels = [inverse_mapping[val[0]] for val in labels]

            output_dict = []
            duration_counter = 0
            i=0
            while duration_counter < nombre_de_minutes * 60 * 50:
                entry = {
                    'label': labels[i],
                    'norm': truncate_generated_signals[i].flatten().tolist()
                }
                output_dict.append(entry)
                duration_counter += len(truncate_generated_signals[i].flatten().tolist())
                i +=1


            # Obtenir la date et l'heure actuelles et les formater pour le nom du fichier
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            
            # Créer le nom du fichier avec la date, l'heure incluses, ainsi que les paramètres choisis
            file_name = f"match_{current_time}_nb_match_{nombre_de_matchs}_duree_{nombre_de_minutes}_style_{style_de_jeu}.json"
            
            # Chemin vers le dossier où vous souhaitez enregistrer le fichier
            folder_path = self.workspace_path+'/output'

            # Chemin complet du fichier à enregistrer
            file_path = os.path.join(folder_path, file_name)

            # Créer le dossier s'il n'existe pas
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Enregistrement du fichier JSON bien indenté
            with open(file_path, 'w') as file:
                json.dump(output_dict, file, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Exécute le générateur de séquences de match avec CGAN.")
    parser.add_argument("--nombre_de_matchs", type=int, default=1, help="Nombre de matchs à générer.")
    parser.add_argument("--nombre_de_minutes", type=int, default=15, help="Durée en minutes du match généré.")
    parser.add_argument("--style_de_jeu", type=str, choices=['Offensif', 'Defensif', 'Equilibré'], default='Equilibré', help="Style de jeu pour le match généré.")
    parser.add_argument("--model_path", type=str, default='generator_model.h5', help="Chemin vers le modèle pré-entraîné.")

    args = parser.parse_args()

    cgan = CGAN()
    if not os.path.exists(args.model_path):
        cgan.train(epochs=30000, batch_size=32, sample_interval=5000)
        cgan.generator.save(args.model_path)
    
    cgan.generate_match(
        nombre_de_match=args.nombre_de_matchs, 
        nombre_de_minutes=args.nombre_de_minutes, 
        style_de_jeu=args.style_de_jeu, 
        model_path=args.model_path)

    # workspace_path = os.getcwd()
    # if not os.path.exists(workspace_path+'/generator_model.h5'):
    #     cgan.train(epochs=30000, batch_size=32, sample_interval=5000)
    #     cgan.generator.save('generator_model.h5')
    
    # cgan.generate_match(nombre_de_matchs=2, nombre_de_minutes=15, style_de_jeu='Offensif', model_path='generator_model.h5')

    

    
    
    


