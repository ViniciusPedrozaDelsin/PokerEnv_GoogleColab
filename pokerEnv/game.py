from treys import Card, Evaluator
import numpy as np
import pygame
import sys
import os

class Game:

    # GUI Constants
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 800
    CARD_WIDTH = 80
    CARD_HEIGHT = 120
    BG_COLOR = (34, 139, 34)
    
    
    def __init__(self):
        # Initialize Pygame
        self.pokerGame = pygame
        
        # Initialize Evaluator
        self.evaluator = Evaluator()
        
        # Game Objects
        self.dealer = None
        self.table = None
        self.players = []
        
        # State Count
        self.STATE = 0

        # End Game
        self.END_GAME = False
        
        # Rotate Game
        self.ROTATE = False
        
        # Game Total Stack
        self.TOTAL_STACK = 0
        
        # Set Verbose
        self.VERBOSE = False
        
        # Set Debbug
        self.DEBBUG = False
    
    # ==================================== Game Objects ====================================
    # Add the Dealer to the Game
    def add_dealer(self, dealer):
        self.dealer = dealer
        
    # Add the Table to the Game
    def add_table(self, table):
        self.table = table
    
    # Add a Player to the Game
    def add_player(self, player):
        self.players.append(player)
    
    
    
    # ==================================== Game self.STATEs ====================================
    # Start Game
    def start(self):
        # Start Poker Game
        self.pokerGame.init()
        
        # Setup Clock
        clock = self.pokerGame.time.Clock()
        clock = self.pokerGame.time
        self.DELAY_TIME = 1

        # Setup display
        self.screen = self.pokerGame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.pokerGame.display.set_caption("Poker GUI")
        
        # Load GUI Images
        self.load_images()
        
        # GUI Cards Informations
        self.setup_cards_positions_GUI()
        
        # GUI Names & Stacks Informations
        self.setup_names_stacks_positions_GUI()
        
        # GUI Names & Stacks Informations
        self.setup_button_positions_GUI()
        
        # Running Flag
        running = True
        
        # Start Time
        START_TIME = 0
        
        while running:
            
            CURRENT_TIME = clock.get_ticks()
            
            # Paint Background
            self.screen.fill(self.BG_COLOR)
            
            # Draw Cards, Names, Stacks & Button
            self.draw_cards_positions_GUI()
            self.draw_table_cards_GUI()
            self.draw_button_GUI()
            
            
            if CURRENT_TIME - START_TIME > self.DELAY_TIME:
                START_TIME = clock.get_ticks()
                self.gameState()

                if self.END_GAME:
                    running = False
                    return

                if self.STATE < 16:
                    self.STATE += 1
                else:
                    self.STATE = 1
                
            for event in self.pokerGame.event.get():
                if event.type == self.pokerGame.QUIT:
                    running = False
                    self.end()
                    sys.exit()
                elif event.type == self.pokerGame.KEYDOWN:
                    if event.key == self.pokerGame.K_q:
                        running = False
                        self.end()
                    elif event.key == self.pokerGame.K_a:
                        self.gameState()
                        if self.STATE < 15:
                            self.STATE += 1
                    elif event.key == self.pokerGame.K_r:
                        self.STATE = 0
                    elif event.key == self.pokerGame.K_s:
                        self.STATE = 1
                        
            self.pokerGame.display.flip()
    
    # End Game
    def end(self):
        self.pokerGame.quit()
    
    def gameState(self):
        
        # Start the Game
        if self.STATE == 0:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            
            self.ROTATE = False
            self.table.hardResetTable()
            positions_to_play = []
            total_stack = 0
            for player in self.players:
                player.resetPlayer()
                #position = self.table.fillPosition()
                positions_to_play.append(self.table.fillPosition())
                
            for player in self.players:
                position = np.random.choice(positions_to_play)
                player.set_table_position(position)
                positions_to_play.remove(position)
                #stack = player.set_random_stack()
                stack = player.set_stack(1000)
                player.stack_last_round = stack
                total_stack += stack
            self.TOTAL_STACK = total_stack
            
        
        # Setup Positions
        elif self.STATE == 1:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            
            self.dealer.resetDealer()
            self.table.softResetTable()
            self.dealer.addTable(self.table)
            self.players.sort(key=lambda p: p.table_position)
            for player in self.players:
                player.resetHand()
                player.all_in = False
                player.all_in_this_round = False
                player.gain_pot_limit = 9999999
                if player.stack != 0:
                    self.dealer.addPlayer(player)
                player.set_player_status(4)
            self.dealer.set_number_of_player()
        
        # Setup Game Rules
        elif self.STATE == 2:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            
            self.dealer.set_minimum_bet(60)
            self.dealer.set_maximum_bet(30000)
            self.GAME_MINIMUM_BET = self.dealer.minimum_bet
            self.GAME_MAXIMUM_BET = self.dealer.maximum_bet
        
        # Set Players Status
        elif self.STATE == 3:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            
            # Redudance
            self.n_of_players_round = 0
            for player in self.players:
                if player.stack != 0:
                    player.set_player_status(0)
                    self.n_of_players_round += 1
                    winner = player.name

                # Calculate Rewards
                if player.model_name != None:
                    player.calculateReward()
                    player.stack_last_round = player.stack

            # End Game
            if self.n_of_players_round <= 1:
                print("=======================")
                print(f"Game Winner {winner}")
                print("=======================")
                self.winner = winner
                self.END_GAME = True
        
        # Dealer Give the Button
        elif self.STATE == 4:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            
            if self.ROTATE == True: 
                self.dealer.rotate_game_positions()
            else:
                self.dealer.set_game_positions()
        
        # Dealer Shuffle
        elif self.STATE == 5:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            
            self.dealer.shuffleCards()
            #self.dealer.cutDeck()
            
        # Dealer Deal
        elif self.STATE == 6:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            
            self.dealer.deal()
            self.table.set_show_cards(True)
        
        # Betting Round Pre-Flop
        elif self.STATE == 7:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            
            self.bettingRound("Pre-Flop")
            
        # Open Flop
        elif self.STATE == 8:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            
            self.dealer.burnCard()
            self.dealer.openFlop()
        
        # Betting Round Flop
        elif self.STATE == 9:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            
            self.bettingRound("Flop")
        
        # Open Turn
        elif self.STATE == 10:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            
            self.dealer.burnCard()
            self.dealer.openTurn()
        
        # Betting Round Turn
        elif self.STATE == 11:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            
            self.bettingRound("Turn")
        
        # Open River
        elif self.STATE == 12:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            
            self.dealer.burnCard()
            self.dealer.openRiver()
        
        # Betting Round River
        elif self.STATE == 13:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            
            self.bettingRound("River")
        
        # Evaluate Hands
        elif self.STATE == 14:
            if self.DEBBUG: print(f"STATE {self.STATE}")
            self.evalFinalHands()
            
        # End Round
        elif self.STATE == 15:
            if self.DEBBUG: print(f"STATE {self.STATE}")
        
            # Debbug
            if self.VERBOSE: print("==================================")
            if self.VERBOSE: print(f"POT AT END GAME: {self.table.pot}")
            
            if self.dealer.round_final_players_order == []:
                winner_name = next(iter(self.dealer.dict_round_position))
                if self.VERBOSE: print(f"Winner: {winner_name}")
                if self.VERBOSE: print(f"JTC Player Available in the hand: {self.dealer.dict_round_position}")
                for player in self.players:
                    if player.name == winner_name:
                        player.increase_stack(self.table.pot)
                        self.table.pot = 0
            else:
                groups = {}
                for entry in self.dealer.round_final_players_order:
                    for player_name, player_and_score in entry.items():
                        if player_and_score[1] not in groups:
                            groups[player_and_score[1]] = []
                        groups[player_and_score[1]].append({player_name: [player_and_score[0], player_and_score[1]]})
                result = [groups[player_and_score[1]] for player_and_score[1] in sorted(groups)]
                if self.VERBOSE: print(f"Results: {result}")
                
                
                while self.table.pot > 0:
                    for group in result:
                        if self.VERBOSE: print(f"Actual Group: {group}")
                        number_of_winners = len(group)
                        group_reach_top_limit = 0
                        while group_reach_top_limit < number_of_winners:
                            gain_for_player = self.table.pot / (number_of_winners-group_reach_top_limit)
                            if self.VERBOSE: print(f"Gain for Player: {gain_for_player}")
                            for dict_player in group:
                                    for k, v in dict_player.items():
                                        if gain_for_player >= v[0].gain_pot_limit:
                                            if self.VERBOSE: print(f"Player: {v[0].name} | Gain: {v[0].gain_pot_limit} || IF")
                                            v[0].increase_stack(v[0].gain_pot_limit)
                                            self.table.decrease_pot(v[0].gain_pot_limit)
                                            v[0].decrease_pot_limit(v[0].gain_pot_limit)
                                            group_reach_top_limit += 1
                                        else:
                                            if self.VERBOSE: print(f"Player: {v[0].name} | Gain: {gain_for_player} || ELSE")
                                            v[0].increase_stack(gain_for_player)
                                            v[0].decrease_pot_limit(gain_for_player)
                                            self.table.decrease_pot(gain_for_player)
                                        if self.VERBOSE: print(f"Table Pot Update: {self.table.pot}")
                                        if round(self.table.pot, 10) == 0:
                                            return
        
        
        # Verification
        elif self.STATE == 16:        
            self.verifyTotalStack()
            
    
    
    def bettingRound(self, round_type):
        
        # Set All Players Rotation Stacks To Zero
        for player in self.players:
            #if player.status == "In Hand Open" or player.status == "In Hand Close":
            player.rotation_stack = 0
            
        more_one_rotation = True
        rotation_number = 0
        current_pot_size = 0
        rotation_index = None
        while more_one_rotation:            
            more_one_rotation = False
            rotation_list = []
            rotation_list = [v[0] for k, v in sorted(self.dealer.dict_round_position.items(), key=lambda item: item[1][0])]
            rotation_list = rotation_list[::-1]
            #print(rotation_list)
            if round_type == "Pre-Flop":
                rotation_list = self.shift_list_relative(2, rotation_list)
                if rotation_number == 0:
                    current_pot_size = self.GAME_MINIMUM_BET
                    for player in self.players:
                        if player.status == "In Hand Open" or player.status == "In Hand Close":
                            if player.round_position_encoded == "SB":
                                if player.stack >= self.GAME_MINIMUM_BET/2:
                                    value = self.GAME_MINIMUM_BET/2
                                else:
                                    value = player.stack
                                player.rotation_stack += value
                                player.decrease_stack(value)
                                self.table.increase_pot(value)
                                self.table.pot_round += value
                            elif player.round_position_encoded == "BB":
                                if player.stack >= self.GAME_MINIMUM_BET:
                                    value = self.GAME_MINIMUM_BET
                                else:
                                    value = player.stack
                                player.rotation_stack += value
                                player.decrease_stack(value)
                                self.table.increase_pot(value)
                                self.table.pot_round += value
            rotation_list_counter = 0
            if rotation_index != None:
                rotation_list = self.shift_list_end_number(rotation_index, rotation_list)
            # Debbug
            #print(f"Rotation Number: {rotation_number}")
            old_current_pot_size = current_pot_size
            while rotation_list != [] and old_current_pot_size == current_pot_size:
                if self.VERBOSE: print("====================================")
                if self.VERBOSE: print(f"==== Current Pot Size {current_pot_size} =======")
                player_to_play = None
                for player in self.players:
                    if player.status == "In Hand Open" or player.status == "In Hand Close":
                        if player.round_position == rotation_list[0]:
                            player_to_play = player
                if self.VERBOSE: print(player_to_play)
                # Check Player Available Actions
                possible_actions_for_this_player = []
                dif_player_pot = current_pot_size - player_to_play.rotation_stack
                if self.VERBOSE: print(f"Player: {player_to_play.name} | Rotation Stack: {player_to_play.rotation_stack}")
                if player_to_play.all_in == False:
                    if dif_player_pot == 0:
                        possible_actions_for_this_player.append("Check")
                        possible_actions_for_this_player.append("Raise2x")
                        possible_actions_for_this_player.append("Raise3x")
                        possible_actions_for_this_player.append("Raise4x")
                        possible_actions_for_this_player.append("Raise5x")
                        possible_actions_for_this_player.append("RaiseHalfPot")
                        possible_actions_for_this_player.append("RaisePot")
                        possible_actions_for_this_player.append("All In")
                    else:
                        possible_actions_for_this_player.append("Call")
                        possible_actions_for_this_player.append("Raise2x")
                        possible_actions_for_this_player.append("Raise3x")
                        possible_actions_for_this_player.append("Raise4x")
                        possible_actions_for_this_player.append("Raise5x")
                        possible_actions_for_this_player.append("RaiseHalfPot")
                        possible_actions_for_this_player.append("RaisePot")
                        possible_actions_for_this_player.append("All In")
                        possible_actions_for_this_player.append("Fold")
                    
                    action = player_to_play.action(
                        round_type,
                        self.table.cards,
                        self.GAME_MINIMUM_BET, 
                        self.GAME_MAXIMUM_BET,
                        current_pot_size,
                        dif_player_pot,
                        self.table.pot,
                        self.n_of_players_round,
                        len(self.dealer.dict_round_position),
                        len(rotation_list),
                        possible_actions_for_this_player
                    )
                    if action[0] == "Check":
                        pass
                    elif action[0] == "Fold":
                        self.dealer.dict_round_position.pop(player_to_play.name)
                        player_to_play.set_player_status(3)
                    elif action[0] == "Call":
                        player_to_play.decrease_stack(action[1])
                        self.table.increase_pot(action[1])
                        self.table.pot_round += action[1]
                        player_to_play.rotation_stack += action[1]
                    elif action[0] == "All In":
                        player_to_play.all_in = True
                        player_to_play.all_in_this_round = True
                        if (player_to_play.stack + player_to_play.rotation_stack) > current_pot_size:
                            more_one_rotation = True
                            current_pot_size = action[1] + player_to_play.rotation_stack
                            rotation_index = player_to_play.round_position
                        self.table.increase_pot(action[1])
                        self.table.pot_round += action[1]
                        player_to_play.rotation_stack += action[1]
                        player_to_play.decrease_stack(action[1])
                    else:
                        more_one_rotation = True
                        self.table.increase_pot(action[1])
                        self.table.pot_round += action[1]
                        player_to_play.rotation_stack += action[1]
                        player_to_play.decrease_stack(action[1])
                        old_current_pot_size = current_pot_size
                        current_pot_size += (action[1] - dif_player_pot)
                        rotation_index = player_to_play.round_position
                
                rotation_list.remove(player_to_play.round_position)
                rotation_list_counter += 1
                
                # One Player Left
                if len(self.dealer.dict_round_position) == 1:
                    more_one_rotation = False
                    #self.STATE = 15
                    self.STATE = 14
                    self.ROTATE = True
                    if self.VERBOSE: print("===== END BETTING ROUND =====")
                    return

            rotation_number += 1
        
        # Check ALL IN
        for player in self.players:
            other_players_rotation_stack = []
            if player.all_in_this_round == True:
                if self.VERBOSE: print(f"================== Player ALL IN: {player} ==================")
                for other_player in self.players:
                    if other_player.name != player.name:
                        if other_player.rotation_stack >= player.rotation_stack:
                            other_players_rotation_stack.append(player.rotation_stack)
                        else:
                            other_players_rotation_stack.append(other_player.rotation_stack)
                            
                player.gain_pot_limit = sum(self.table.pot_round_list) + player.rotation_stack + sum(other_players_rotation_stack)
                
                if self.VERBOSE: print(f"================== OTHERS ROUND STACK: {other_players_rotation_stack} ==================")
                if self.VERBOSE: print(f"================== SUM OTHERS ROUND STACK: {sum(other_players_rotation_stack)} ==================")
                if self.VERBOSE: print(f"================== PLAYER ROUND STACK: {player.rotation_stack} ==================")
                if self.VERBOSE: print(f"================== SUM TABLE POT: {sum(self.table.pot_round_list)} ==================")
                if self.VERBOSE: print(f"================== Player GAIN POT LIMIT: {player.gain_pot_limit} ==================")
                player.all_in_this_round = False
        
        self.table.pot_round_list.append(self.table.pot_round)
        if self.VERBOSE: print(f"Gold Added in the Pot in this round: {self.table.pot_round}")
        self.table.pot_round = 0
        if self.VERBOSE: print("===== END BETTING ROUND =====")
        self.ROTATE = True
    
    
    # Evaluate Final Hands
    def evalFinalHands(self):
        #print(self.dealer.dict_round_position)
        
        # Encoding Board Cards
        board_cards = []
        for card in self.table.cards:
            board_cards.append(Card.new(self.dealer.encodingCards(card)))
        
        players_hands = {}
        for k, v in self.dealer.dict_round_position.items():
            for player in self.players:
                if player.name == k:
                    players_hands[k] = [player, player.hand_encoded]
        
        finals_players_order = []
        for k, v in players_hands.items():
            player_cards = [Card.new(v[1][0]), Card.new(v[1][1])]
            if self.VERBOSE: print(player_cards)
            score = self.evaluator.evaluate(board_cards, player_cards)
            if self.VERBOSE: print(f"Player: {k} ||| Score: {score}")
            finals_players_order.append({k: [v[0], score]})
        finals_players_order = sorted(finals_players_order, key=lambda x: list(x.values())[0][1])
        self.dealer.round_final_players_order = finals_players_order
    
    
    # ==================================== Auxiliary Function ====================================
   
    
    def shift_list_end_number(self, end_index, lst):
        while lst[-1] != end_index:
            lst = self.shift_list_relative(1, lst)
        return lst[:-1]
    
    def shift_list_relative(self, relative_index, lst):
        return lst[relative_index:] + lst[:relative_index]
        
    
    def verifyTotalStack(self):
        total_stack = 0
        for player in self.players:
            total_stack += player.stack
        
        if total_stack != self.TOTAL_STACK:
            if self.VERBOSE: print(f"Total Stack Error: {total_stack}")
            sys.exit()
        else:
            if self.VERBOSE: print(f"{total_stack}, {self.TOTAL_STACK}")
            if self.DEBBUG: print("=======================")
            if self.DEBBUG: print("Total Stack Verified!")
            if self.DEBBUG: print("=======================")
        
        
    
    # ==================================== GUI Methods ====================================
    # Load Game Images
    def load_images(self):
        self.cards_images = self.load_card_images()
        self.CARD_BACKGROUND = self.load_card_bg("pokerEnv/images/BG.png")
        self.NO_CARD_BACKGROUND = self.load_card_bg("pokerEnv/images/BG_noCard.png")
        self.FOLDED_CARD_BACKGROUND = self.load_card_bg("pokerEnv/images/BG_foldedCard.png")
        self.FLIPPED_CARD_BACKGROUND = self.load_card_bg("pokerEnv/images/BG_flippedCard.png")
        self.BTN_IMAGE = self.load_btn()
    
    # Load Card Images
    def load_card_images(self, folder="pokerEnv/images"):
        cards_images = {}
        for suit in ['h', 'd', 'c', 's']:
            for rank in ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']:
                name = rank + suit
                path = os.path.join(folder, name + ".png")
                image = pygame.image.load(path)
                image = pygame.transform.scale(image, (self.CARD_WIDTH, self.CARD_HEIGHT))
                cards_images[name] = image
        return cards_images
    
    # Load Card Background
    def load_card_bg(self, folder):
        image_card_bg = pygame.image.load(folder)
        image_card_bg = pygame.transform.scale(image_card_bg, (self.CARD_WIDTH, self.CARD_HEIGHT))
        return image_card_bg
    
    # Load Dealer Button
    def load_btn(self, folder="pokerEnv/images/BTN.png"):
        image_btn = pygame.image.load(folder)
        image_btn = pygame.transform.scale(image_btn, (25, 25))
        return image_btn
    
    # Setup Cards Positions
    def setup_cards_positions_GUI(self):
        if self.table.number_of_positions == 10:   
        
            self.GUI_PLAYER_CARDS_POSITIONS = {
                0: [(160, 630), (260, 630)], 
                1: [(410, 630), (510, 630)], 
                2: [(660, 630), (760, 630)], 
                3: [(800, 430), (900, 430)], 
                4: [(800, 230), (900, 230)], 
                5: [(660, 60), (760, 60)], 
                6: [(410, 60), (510, 60)], 
                7: [(160, 60), (260, 60)], 
                8: [(20, 250), (120, 250)], 
                9: [(20, 440), (120, 440)]
            }
            
            self.GUI_TABLE_CARDS_POSITIONS = {
                0: (260, 320),
                1: (360, 320),
                2: (460, 320),
                3: (560, 320),
                4: (660, 320)
            }
    
    # Setup Names and Stacks Positions
    def setup_names_stacks_positions_GUI(self):
        if self.table.number_of_positions == 10:
            
            self.GUI_PLAYER_NAME_STACK_POSITIONS = {
                0: [(160, 760), (260, 760)],
                1: [(410, 760), (510, 760)],
                2: [(660, 760), (760, 760)],
                3: [(800, 560), (900, 560)],
                4: [(800, 360), (900, 360)],
                5: [(660, 30), (760, 30)],
                6: [(410, 30), (510, 30)],
                7: [(160, 30), (260, 30)],
                8: [(20, 220), (120, 220)],
                9: [(20, 410), (120, 410)]
            }
            
            self.GUI_TABLE_NAME_STACK_POSITIONS = {
                0: (480, 480)
            }
    
    # Setup Button Positions
    def setup_button_positions_GUI(self):
        if self.table.number_of_positions == 10:
            
            self.GUI_BUTTON_POSITIONS = {
                0: (240, 590),
                1: (470, 590),
                2: (730, 590),
                3: (770, 470),
                4: (770, 280),
                5: (740, 190),
                6: (490, 190),
                7: (240, 190),
                8: (220, 280),
                9: (220, 470)
            }
    
    # Draw Cards Names and Stacks Positions
    def draw_cards_positions_GUI(self):
        
        # Players Cards
        for n_pos in list(range(self.table.number_of_positions)):
            for player in self.players:
                if player.table_position == n_pos:
                    if player.status == "In Hand Open" and self.STATE > 7:
                        # First Card
                        self.screen.blit(self.CARD_BACKGROUND, self.GUI_PLAYER_CARDS_POSITIONS[player.table_position][0])
                        self.screen.blit(self.cards_images[player.hand_encoded[0]], self.GUI_PLAYER_CARDS_POSITIONS[player.table_position][0])
                        
                        # Second Card
                        self.screen.blit(self.CARD_BACKGROUND, self.GUI_PLAYER_CARDS_POSITIONS[player.table_position][1])
                        self.screen.blit(self.cards_images[player.hand_encoded[1]], self.GUI_PLAYER_CARDS_POSITIONS[player.table_position][1])
                        
                        # Name & Stack
                        self.screen.blit(self.pokerGame.font.SysFont(None, 25).render(str(player.name), True, (255, 255, 0)), self.GUI_PLAYER_NAME_STACK_POSITIONS[player.table_position][0])
                        self.screen.blit(self.pokerGame.font.SysFont(None, 25).render(str(player.stack), True, (255, 255, 0)), self.GUI_PLAYER_NAME_STACK_POSITIONS[player.table_position][1])
                    
                    elif player.status == "In Hand Close" and self.STATE > 7:
                        # First Card
                        self.screen.blit(self.FLIPPED_CARD_BACKGROUND, self.GUI_PLAYER_CARDS_POSITIONS[player.table_position][0])
                        
                        # Second Card
                        self.screen.blit(self.FLIPPED_CARD_BACKGROUND, self.GUI_PLAYER_CARDS_POSITIONS[player.table_position][1])
                        
                        # Name & Stack
                        self.screen.blit(self.pokerGame.font.SysFont(None, 25).render(str(player.name), True, (255, 255, 0)), self.GUI_PLAYER_NAME_STACK_POSITIONS[player.table_position][0])
                        self.screen.blit(self.pokerGame.font.SysFont(None, 25).render(str(player.stack), True, (255, 255, 0)), self.GUI_PLAYER_NAME_STACK_POSITIONS[player.table_position][1])
                    
                    elif player.status == "Not in Hand":
                        # First Card
                        self.screen.blit(self.NO_CARD_BACKGROUND, self.GUI_PLAYER_CARDS_POSITIONS[player.table_position][0])
                        
                        # Second Card
                        self.screen.blit(self.NO_CARD_BACKGROUND, self.GUI_PLAYER_CARDS_POSITIONS[player.table_position][1])
                        
                        # Name & Stack
                        self.screen.blit(self.pokerGame.font.SysFont(None, 25).render(str(player.name), True, (255, 255, 0)), self.GUI_PLAYER_NAME_STACK_POSITIONS[player.table_position][0])
                        self.screen.blit(self.pokerGame.font.SysFont(None, 25).render(str(player.stack), True, (255, 255, 0)), self.GUI_PLAYER_NAME_STACK_POSITIONS[player.table_position][1])
                    
                    elif player.status == "Folded":
                        # First Card
                        self.screen.blit(self.FOLDED_CARD_BACKGROUND, self.GUI_PLAYER_CARDS_POSITIONS[player.table_position][0])
                        
                        # Second Card
                        self.screen.blit(self.FOLDED_CARD_BACKGROUND, self.GUI_PLAYER_CARDS_POSITIONS[player.table_position][1])
                        
                        # Name & Stack
                        self.screen.blit(self.pokerGame.font.SysFont(None, 25).render(str(player.name), True, (255, 255, 0)), self.GUI_PLAYER_NAME_STACK_POSITIONS[player.table_position][0])
                        self.screen.blit(self.pokerGame.font.SysFont(None, 25).render(str(player.stack), True, (255, 255, 0)), self.GUI_PLAYER_NAME_STACK_POSITIONS[player.table_position][1])
                    
                    else:
                        for i in list(range(0, 2)):
                            self.screen.blit(self.NO_CARD_BACKGROUND, self.GUI_PLAYER_CARDS_POSITIONS[player.table_position][i])
        
        # Draw No Card in Empty Spaces
        for available_sit in self.table.available_positions:
            for i in list(range(0, 2)):
                self.screen.blit(self.NO_CARD_BACKGROUND, self.GUI_PLAYER_CARDS_POSITIONS[available_sit][i])
            
    
    # Draw Table Cards
    def draw_table_cards_GUI(self):
        if self.table.cards == [] and self.table.show_cards == False:
            for i in list(range(0, 5)):
                self.screen.blit(self.NO_CARD_BACKGROUND, self.GUI_TABLE_CARDS_POSITIONS[i])
        
        elif self.table.cards == [] and self.table.show_cards == True:
            # Draw the Pot size
            self.screen.blit(self.pokerGame.font.SysFont(None, 25).render(str(self.table.pot), True, (255, 255, 0)), self.GUI_TABLE_NAME_STACK_POSITIONS[0])
            for i in list(range(0, 5)):
                self.screen.blit(self.FLIPPED_CARD_BACKGROUND, self.GUI_TABLE_CARDS_POSITIONS[i])
                
        else:
            # Draw the Pot size
            self.screen.blit(self.pokerGame.font.SysFont(None, 25).render(str(self.table.pot), True, (255, 255, 0)), self.GUI_TABLE_NAME_STACK_POSITIONS[0])
            
            table_cards_count = 0
            for card in self.table.cards_encoded:
                self.screen.blit(self.CARD_BACKGROUND, self.GUI_TABLE_CARDS_POSITIONS[table_cards_count])
                self.screen.blit(self.cards_images[card], self.GUI_TABLE_CARDS_POSITIONS[table_cards_count])
                table_cards_count += 1
            
            while table_cards_count < 5:
                self.screen.blit(self.FLIPPED_CARD_BACKGROUND, self.GUI_TABLE_CARDS_POSITIONS[table_cards_count])
                table_cards_count += 1
    
    # Draw Dealer Button
    def draw_button_GUI(self):
        # Dealer Button
        only_two_players = True
        for player in self.players:
            if player.status != "Away":
                if player.round_position_encoded == "BTN":
                    self.screen.blit(self.BTN_IMAGE, self.GUI_BUTTON_POSITIONS[player.table_position])
                    only_two_players = False
        
        # Dealer Button in SB Position     
        for player in self.players:
            if player.status != "Away" and only_two_players == True:
                if player.round_position_encoded == "SB":
                    self.screen.blit(self.BTN_IMAGE, self.GUI_BUTTON_POSITIONS[player.table_position])