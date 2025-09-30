import random

class Deck:
    def __init__(self):
        self.number_of_cards = 52
        self.deck_of_cards = list(range(0, self.number_of_cards))
    
    # Mapping every Card to a Number
    def mappingCards(self, value, direction):
        cards_numbers_list = [
            (0, "2d"),
            (1, "3d"),
            (2, "4d"),
            (3, "5d"),
            (4, "6d"),
            (5, "7d"),
            (6, "8d"),
            (7, "9d"),
            (8, "Td"),
            (9, "Jd"),
            (10, "Qd"),
            (11, "Kd"),
            (12, "Ad"),
            (13, "2s"),
            (14, "3s"),
            (15, "4s"),
            (16, "5s"),
            (17, "6s"),
            (18, "7s"),
            (19, "8s"),
            (20, "9s"),
            (21, "Ts"),
            (22, "Js"),
            (23, "Qs"),
            (24, "Ks"),
            (25, "As"),
            (26, "2h"),
            (27, "3h"),
            (28, "4h"),
            (29, "5h"),
            (30, "6h"),
            (31, "7h"),
            (32, "8h"),
            (33, "9h"),
            (34, "Th"),
            (35, "Jh"),
            (36, "Qh"),
            (37, "Kh"),
            (38, "Ah"),
            (39, "2c"),
            (40, "3c"),
            (41, "4c"),
            (42, "5c"),
            (43, "6c"),
            (44, "7c"),
            (45, "8c"),
            (46, "9c"),
            (47, "Tc"),
            (48, "Jc"),
            (49, "Qc"),
            (50, "Kc"),
            (51, "Ac")
        ]
        
        if direction == True:
            cards = dict(cards_numbers_list)
            output = cards[value]
        else:
            numbers = {v: k for k, v in cards_numbers_list}
            output = numbers[value]
        
        return output
        
    
    # Number to Card
    def encodingCards(self, number):
        card = self.mappingCards(number, True)
        return card
    
    # Card to Number
    def decodingCards(self, card):
        number = self.mappingCards(card, False)
        return number
    
    # Get Deck of Cards
    def get_deck_of_cards(self):
        return self.deck_of_cards
    
    # Reset Deck
    def resetDeck(self):
        self.deck_of_cards = list(range(0, self.number_of_cards))
        
    # Shuffle Cards
    def shuffleCards(self):
        random.shuffle(self.deck_of_cards)
    
    # Remove Card
    def removeCardFromDeck(self, card):
        if type(card) == str:
            card = self.decodingCards(card)
        self.deck_of_cards.remove(card)
    
    # Get the Top Card of the Deck
    def get_top_card(self):
        top_card = self.deck_of_cards[0]
        self.removeCardFromDeck(top_card)
        return top_card