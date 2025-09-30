import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from collections import deque
import random
import numpy as np
import os

class Player:
    
    POSSIBLE_STATUS = ('In Hand Open', 'In Hand Close', 'Not in Hand', 'Folded', 'Away')
    POSSIBLE_ACTIONS = ('Check', 'Fold', 'Call', 'Raise2x', 'Raise3x', 'Raise4x', 'Raise5x', 'RaiseHalfPot', 'RaisePot', 'All In')
    POSSIBLE_ROUNDS = ('Pre-Flop', 'Flop', 'Turn', 'River')
    
    # Possible Positions
    POSSIBLE_POSITIONS = {
        10 : ('BTN', 'CO', 'HJ', 'MP2', 'MP1', 'UTG+2', 'UTG+1', 'UTG', 'BB', 'SB'),
        9 : ('BTN', 'CO', 'HJ', 'MP1', 'UTG+2', 'UTG+1', 'UTG', 'BB', 'SB'),
        8 : ('BTN', 'CO', 'HJ', 'UTG+2', 'UTG+1', 'UTG', 'BB', 'SB'),
        7 : ('BTN', 'CO', 'HJ', 'UTG+1', 'UTG', 'BB', 'SB'),
        6 : ('BTN', 'CO', 'HJ', 'UTG', 'BB', 'SB'),
        5 : ('BTN', 'CO', 'UTG', 'BB', 'SB'),
        4 : ('BTN', 'UTG', 'BB', 'SB'),
        3 : ('BTN', 'BB', 'SB'),
        2 : ('BB', 'SB')
    }
    
    def __init__(self, name, human=False, model_name=None, update_model=False):
        self.name = name
        self.human = human
        self.model_name = model_name
        self.status = self.POSSIBLE_STATUS[4]
        self.table_position = None
        self.round_position = None
        self.round_position_encoded = None
        self.hand = None
        self.hand_encoded = None
        self.stack = 0
        self.stack_last_round = 0
        self.rotation_stack = 0
        self.gain_pot_limit = 9999999
        self.all_in = False
        self.all_in_this_round = False

        # Neural Network Reiforcement Learning Variables
        self.gamma = 0.999
        self.epsilon = 0
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.9975
        self.batch_size = 32
        self.window_size = 30
        self.update_model = update_model
        self.save_models_steps = 200
        self.train_counter = 0
        if self.model_name != None:
            folder_name = "pokerEnv/models/"
            if os.path.isdir(f"{folder_name}{self.model_name}"):
                self.model = {}
                self.model['Pre-Flop'] = tf.keras.models.load_model(f"{folder_name}{model_name}/{model_name}_preflop.keras")
                self.model['Flop'] = tf.keras.models.load_model(f"{folder_name}{model_name}/{model_name}_flop.keras")
                self.model['Turn'] = tf.keras.models.load_model(f"{folder_name}{model_name}/{model_name}_turn.keras")
                self.model['River'] = tf.keras.models.load_model(f"{folder_name}{model_name}/{model_name}_river.keras")
            else:
                model = self.modelBuild(4, len(self.POSSIBLE_ACTIONS))
                self.model = {}
                self.model['Pre-Flop'] = model['Pre-Flop']
                self.model['Flop'] = model['Flop']
                self.model['Turn'] = model['Turn']
                self.model['River'] = model['River']

        # Memory Variables
        self.memory_lenght = 8192
        self.memory = {}
        self.memory['Pre-Flop'] = deque(maxlen=self.memory_lenght)
        self.memory['Flop'] = deque(maxlen=self.memory_lenght)
        self.memory['Turn'] = deque(maxlen=self.memory_lenght)
        self.memory['River'] = deque(maxlen=self.memory_lenght)
        self.mem_warmup_steps = {}
        self.mem_warmup_steps['Pre-Flop'] = 4096
        self.mem_warmup_steps['Flop'] = 1024
        self.mem_warmup_steps['Turn'] = 256
        self.mem_warmup_steps['River'] = 128
        
    
    # Print Output
    def __str__(self):
        return f"Name: {self.name} | Status: {self.status} | Table-Position: {self.table_position} | Hand: {self.hand_encoded} | Round-Position {self.round_position},{self.round_position_encoded} | Actual-Stack {self.stack} | Gain-Pot-Limit {self.gain_pot_limit}"
    
    # Reset Player
    def resetPlayer(self):
        self.status = self.POSSIBLE_STATUS[4]
        self.table_position = None
        self.round_position = None
        self.round_position_encoded = None
        self.hand = None
        self.hand_encoded = None
        self.stack = 0
        self.stack_last_round = 0
        self.rotation_stack = 0
        self.gain_pot_limit = 9999999
        self.all_in = False
        self.all_in_this_round = False
    
    # Reset Hand
    def resetHand(self):
        self.hand = None
        self.hand_encoded = None
    
    # Set Status
    def set_player_status(self, status_number):
        self.status = self.POSSIBLE_STATUS[status_number]
        
    # Set Player Table Position
    def set_table_position(self, position):
        self.table_position = position
    
    # Set Player Round Position
    def set_round_position(self, number, max_position):
        self.round_position = number
        self.round_position_encoded = self.POSSIBLE_POSITIONS[max_position][number]
        return [self.name, self.round_position, self.round_position_encoded]
    
    # Set Player Hand
    def set_hand(self, first_card, second_card):
        self.hand = (first_card, second_card)
    
    # Set Player Hand Encoded
    def set_hand_encoded(self, first_card, second_card):
        self.hand_encoded = (first_card, second_card)
        
    # Set Stack
    def set_stack(self, value):
        self.stack = value
        return self.stack
        
    # Set Random Stack
    def set_random_stack(self, values=[200, 1000]):
        self.stack = np.random.randint(values[0], values[1])
        return self.stack
    
    # Increase Stack
    def increase_stack(self, value):
        self.stack += value
    
    # Decrease Stack
    def decrease_stack(self, value):
        if self.stack >= value:
            self.stack -= value
        else:
            self.stack = 0
    
    # Gain Pot Limit
    def increase_pot_limit(self, value):
        self.gain_pot_limit += value
        
    # Decrease Pot Limit
    def decrease_pot_limit(self, value):
        if self.gain_pot_limit >= value:
            self.gain_pot_limit -= value
        else:
            self.gain_pot_limit = 0
    
    def action(self, round_type, board_cards, minimum_bet, maximum_bet, current_pot, dif_current_pot, pot_size, players_in_the_round, players_in_the_hand, left_to_play, possible_actions):
        if self.human == True:
            action = str(input("Action: "))
        elif self.model_name != None:
            inputs = [pot_size, dif_current_pot, left_to_play, players_in_the_hand]
            predictions = self.modelPrediction(self.hand, board_cards, players_in_the_round, self.round_position, inputs, round_type)[0]
            probabilities = tf.nn.softmax(predictions).numpy()
            indices = list(range(len(predictions)))
            index_choice = random.choices(indices, weights=probabilities, k=1)[0]
            q_values = predictions[index_choice]
            action = self.POSSIBLE_ACTIONS[index_choice]
            #print(f"1: Round type: {round_type}, Action: {action}")

            #q_value = np.max(predictions)
            #index = np.argmax(predictions)
            #action = self.POSSIBLE_ACTIONS[max_index]

            # Correction impossible choices
            action = "Fold" if action == "Check" and "Check" not in possible_actions else action
            action = "Check" if action == "Fold" and "Fold" not in possible_actions else action
            #print(f"2: Round type: {round_type}, Action: {action}")

            self.memory[round_type].append([[self.hand, board_cards, players_in_the_round, self.round_position, inputs], [index_choice, q_values], None])

            if np.random.rand() > 0.5:
                self.modelTrain(round_type)
        else:
            action = np.random.choice(possible_actions)
        
        #print(f"{self.name} | Old: {action}")
        
        value = 0
        if action == "Check":
            value = 0
        elif action == "Fold":
            value = 0
        elif action == "Call":
            value = dif_current_pot
        elif action == "Raise2x":
            value = 2*minimum_bet
        elif action == "Raise3x":
            value = 3*minimum_bet
        elif action == "Raise4x":
            value = 4*minimum_bet
        elif action == "Raise5x":
            value = 5*minimum_bet
        elif action == "RaiseHalfPot":
            value = pot_size/2
        elif action == "RaisePot":
            value = pot_size
        elif action == "All In":
            value = self.stack
        
        if self.human == True: value = int(input("Value: "))
        
        if value <= dif_current_pot and action != "Check" and action != "Fold":
            action = "Call"
            value = dif_current_pot
        
        if value >= self.stack:
            action = "All In"
            value = self.stack
        
        #print(f"{self.name} | New: {action} | Value: {value}")

        return [action, value]


    # ============================================= Neural Network RL Model =============================================

    def calculateReward(self):
        for round_type in self.POSSIBLE_ROUNDS:
            for men in self.memory[round_type]:
                if men[2] == None:
                    reward = self.stack - self.stack_last_round
                    men[2] = reward
                    #print(f"Mudou de None para {reward}")

        #print("============ XXXXXXX ===========")
        #print(f"Memory: {self.memory}")
        #print("============ XXXXXXX ===========")

    def modelBuild(self, n_inputs=1, n_outputs=1):
        model_dict = {}
        model_dict['Pre-Flop'] = self.modelBuildRounds(n_inputs, n_outputs, 'Pre-Flop')
        model_dict['Flop'] = self.modelBuildRounds(n_inputs, n_outputs, 'Flop')
        model_dict['Turn'] = self.modelBuildRounds(n_inputs, n_outputs, 'Turn')
        model_dict['River'] = self.modelBuildRounds(n_inputs, n_outputs, 'River')
        return model_dict

    def modelBuildRounds(self, n_inputs=1, n_outputs=1, round_name=None):
        inputs = Input(shape=(n_inputs,), dtype='float32', name='inputs')

        # Cards embedding layer
        cards_embedding_dim = 6
        card_1 = Input(shape=(1,), dtype='int32', name='card_1')
        card_2 = Input(shape=(1,), dtype='int32', name='card_2')
        card_1_emb = layers.Embedding(input_dim=52, output_dim=cards_embedding_dim, embeddings_initializer='glorot_uniform', name='card_1_embedding')(card_1)
        card_2_emb = layers.Embedding(input_dim=52, output_dim=cards_embedding_dim, embeddings_initializer='glorot_uniform', name='card_2_embedding')(card_2)
        card_1_emb = layers.Flatten()(card_1_emb)
        card_2_emb = layers.Flatten()(card_2_emb)

        if round_name == 'Flop' or round_name == 'Turn' or round_name == 'River':
            # Flop cards embedding layer
            card_flop_1 = Input(shape=(1,), dtype='int32', name='card_flop_1')
            card_flop_2 = Input(shape=(1,), dtype='int32', name='card_flop_2')
            card_flop_3 = Input(shape=(1,), dtype='int32', name='card_flop_3')
            card_flop_1_emb = layers.Embedding(input_dim=52, output_dim=cards_embedding_dim, embeddings_initializer='glorot_uniform', name='card_flop_1_embedding')(card_flop_1)
            card_flop_2_emb = layers.Embedding(input_dim=52, output_dim=cards_embedding_dim, embeddings_initializer='glorot_uniform', name='card_flop_2_embedding')(card_flop_2)
            card_flop_3_emb = layers.Embedding(input_dim=52, output_dim=cards_embedding_dim, embeddings_initializer='glorot_uniform', name='card_flop_3_embedding')(card_flop_3)
            card_flop_1_emb = layers.Flatten()(card_flop_1_emb)
            card_flop_2_emb = layers.Flatten()(card_flop_2_emb)
            card_flop_3_emb = layers.Flatten()(card_flop_3_emb)

        if round_name == 'Turn' or round_name == 'River':
            # Turn card embedding layer
            card_turn_1 = Input(shape=(1,), dtype='int32', name='card_turn_1')
            card_turn_1_emb = layers.Embedding(input_dim=52, output_dim=cards_embedding_dim, embeddings_initializer='glorot_uniform', name='card_turn_1_embedding')(card_turn_1)
            card_turn_1_emb = layers.Flatten()(card_turn_1_emb)

        if round_name == 'River':
            # River card embedding layer
            card_river_1 = Input(shape=(1,), dtype='int32', name='card_river_1')
            card_river_1_emb = layers.Embedding(input_dim=52, output_dim=cards_embedding_dim, embeddings_initializer='glorot_uniform', name='card_river_1_embedding')(card_river_1)
            card_river_1_emb = layers.Flatten()(card_river_1_emb)

        # Number of player in the round
        cards_embedding_dim = 4
        npr = Input(shape=(1,), dtype='int32', name='npr')
        npr_emb = layers.Embedding(input_dim=10, output_dim=cards_embedding_dim, embeddings_initializer='glorot_uniform', name='npr_embedding')(npr)
        npr_emb = layers.Flatten()(npr_emb)

        # Player position
        pp_embedding_dim = 4
        pp = Input(shape=(1,), dtype='int32', name='pp')
        pp_emb = layers.Embedding(input_dim=10, output_dim=pp_embedding_dim, embeddings_initializer='glorot_uniform', name='pp_embedding')(pp)
        pp_emb = layers.Flatten()(pp_emb)

        if round_name == 'Pre-Flop':
            x = layers.Concatenate()([card_1_emb, card_2_emb, npr_emb, pp_emb, inputs])
        elif round_name == 'Flop':
            x = layers.Concatenate()([card_1_emb, card_2_emb, card_flop_1, card_flop_2, card_flop_3, npr_emb, pp_emb, inputs])
        elif round_name == 'Turn':
            x = layers.Concatenate()([card_1_emb, card_2_emb, card_flop_1, card_flop_2, card_flop_3, card_turn_1, npr_emb, pp_emb, inputs])
        elif round_name == 'River':
            x = layers.Concatenate()([card_1_emb, card_2_emb, card_flop_1, card_flop_2, card_flop_3, card_turn_1, card_river_1, npr_emb, pp_emb, inputs])

        # MLP head (64 -> 128 -> 64 -> 32)
        x = layers.Dense(32, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Dense(64, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Dense(32, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Dense(16, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        out = layers.Dense(n_outputs, activation='linear', name='q_values')(x)

        if round_name == 'Pre-Flop':
            model = Model(inputs=[card_1, card_2, npr, pp, inputs], outputs=out)
        elif round_name == 'Flop':
            model = Model(inputs=[card_1, card_2, card_flop_1, card_flop_2, card_flop_3, npr, pp, inputs], outputs=out)
        elif round_name == 'Turn':
            model = Model(inputs=[card_1, card_2, card_flop_1, card_flop_2, card_flop_3, card_turn_1, npr, pp, inputs], outputs=out)
        elif round_name == 'River':
            model = Model(inputs=[card_1, card_2, card_flop_1, card_flop_2, card_flop_3, card_turn_1, card_river_1, npr, pp, inputs], outputs=out)

        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def modelPrediction(self, player_cards, board_cards, number_players_remaining, player_position, model_inputs, round_name=None):
        if np.random.rand() < self.epsilon:
            predictions = [np.random.rand(10)]
        else:
            if round_name == 'Pre-Flop':
                X = [np.array([player_cards[0]]), np.array([player_cards[1]]), np.array([number_players_remaining]), np.array([player_position]), np.array([model_inputs])]
            elif round_name == 'Flop':
                X = [np.array([player_cards[0]]), np.array([player_cards[1]]), np.array([board_cards[0]]), np.array([board_cards[1]]), np.array([board_cards[2]]), np.array([number_players_remaining]), np.array([player_position]), np.array([model_inputs])]
            elif round_name == 'Turn':
                X = [np.array([player_cards[0]]), np.array([player_cards[1]]), np.array([board_cards[0]]), np.array([board_cards[1]]), np.array([board_cards[2]]), np.array([board_cards[3]]), np.array([number_players_remaining]), np.array([player_position]), np.array([model_inputs])]
            elif round_name == 'River':
                X = [np.array([player_cards[0]]), np.array([player_cards[1]]), np.array([board_cards[0]]), np.array([board_cards[1]]), np.array([board_cards[2]]), np.array([board_cards[3]]), np.array([board_cards[4]]), np.array([number_players_remaining]), np.array([player_position]), np.array([model_inputs])]
            predictions = self.model[round_name].predict(X, verbose=0)
        return predictions

    def modelTrain(self, round_type):
        # Return if Memory < Batch_Size
        if len(self.memory[round_type]) < (self.batch_size + self.window_size):
            #print(f"Deixou: Round Type: {round_type} | Men Len: {len(self.memory[round_type])}")
            return
        
        # Return util memory warmup
        if len(self.memory[round_type]) < self.mem_warmup_steps[round_type]:
            #print(f"Round Type: {round_type} | Men Len: {len(self.memory[round_type])}")
            return

        if self.train_counter < 600:
            times = 1
        elif self.train_counter >= 600 and self.train_counter < 1000:
            times = 1
        elif self.train_counter >= 1000 and self.train_counter < 10000:
            times = 1
        else:
            times = 2

        for _ in range(times):
            minibatch = random.sample(list(self.memory[round_type])[:-self.window_size], self.batch_size)
            
            X = []
            y = []
            #predictions = self.modelPrediction(self.hand, board_cards, players_in_the_round, self.round_position, inputs, round_type)
            #self.memory[round_type].append([[self.hand, board_cards, players_in_the_round, self.round_position, inputs], [max_index, max_value], None])
            for sample_choice in minibatch:

                # Current network prediction
                q_values = self.modelPrediction(sample_choice[0][0], sample_choice[0][1], sample_choice[0][2], sample_choice[0][3], sample_choice[0][4], round_type)  
                #print(f"ALL: {q_values}")
                reward = sample_choice[2]
                target = sample_choice[1][1] + ((1-self.gamma) * sample_choice[1][1] * (reward/1000))
                target = reward
                #print(f"New Target {target}")

                q_values[0][sample_choice[1][0]] = target

                X.append(sample_choice[0])
                y.append(q_values[0])

            # List of the Inputs
            cards_list = [sublist[0] for sublist in X]
            card_1_list = [sublist[0] for sublist in cards_list]
            card_2_list = [sublist[1] for sublist in cards_list]
            board_list = [sublist[1] for sublist in X]
            npr_list = [sublist[2] for sublist in X]
            pp_list = [sublist[3] for sublist in X]
            inputs_list = [sublist[4] for sublist in X]

            if round_type == 'Flop' or round_type == 'Turn' or round_type == 'River':
                card_board_1_list = [sublist[0] for sublist in board_list]
                card_board_2_list = [sublist[1] for sublist in board_list]
                card_board_3_list = [sublist[2] for sublist in board_list]

            if round_type == 'Turn' or round_type == 'River':
                card_board_4_list = [sublist[3] for sublist in board_list]

            if round_type == 'River':
                card_board_5_list = [sublist[4] for sublist in board_list]

            # Set the learning rate to 0.00001
            #self.model.optimizer.learning_rate.assign(1e-5)

            if round_type == 'Pre-Flop':
                print("PREFLOP: Executou Pre-Flop model Fit")
                self.model['Pre-Flop'].fit([np.array(card_1_list), np.array(card_2_list), np.array(npr_list), np.array(pp_list), np.array(inputs_list)], np.array(y), epochs=1, verbose=0)
            elif round_type == 'Flop':
                print("FLOP: Executou Flop model Fit")
                self.model['Flop'].fit([np.array(card_1_list), np.array(card_2_list), np.array(card_board_1_list), np.array(card_board_2_list), np.array(card_board_3_list), np.array(npr_list), np.array(pp_list), np.array(inputs_list)], np.array(y), epochs=1, verbose=0)
            elif round_type == 'Turn':
                print("Turn: Executou Turn model Fit")
                self.model['Turn'].fit([np.array(card_1_list), np.array(card_2_list), np.array(card_board_1_list), np.array(card_board_2_list), np.array(card_board_3_list), np.array(card_board_4_list), np.array(npr_list), np.array(pp_list), np.array(inputs_list)], np.array(y), epochs=1, verbose=0)
            elif round_type == 'River':
                print("River: Executou River model Fit")
                self.model['River'].fit([np.array(card_1_list), np.array(card_2_list), np.array(card_board_1_list), np.array(card_board_2_list), np.array(card_board_3_list), np.array(card_board_4_list), np.array(card_board_5_list), np.array(npr_list), np.array(pp_list), np.array(inputs_list)], np.array(y), epochs=1, verbose=0)

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Save Models
            if self.train_counter % self.save_models_steps == 0 and self.update_model == True:
                self.saveModels()

            self.train_counter += 1

    def saveModels(self):
        print("Saving Models...")
        self.model['Pre-Flop'].save(f"pokerEnv/models/model_1/{self.model_name}_preflop.keras")
        self.model['Flop'].save(f"pokerEnv/models/model_1/{self.model_name}_flop.keras")
        self.model['Turn'].save(f"pokerEnv/models/model_1/{self.model_name}_turn.keras")
        self.model['River'].save(f"pokerEnv/models/model_1/{self.model_name}_river.keras")