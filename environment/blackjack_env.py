"""
environment/blackjack_env.py — MDP del Blackjack implementado desde cero

Estados: (suma_jugador, carta_dealer, as_utilizable)
Acciones: 0 = PLANTARSE (Stand), 1 = PEDIR (Hit)
Recompensas: +1 ganar, -1 perder o pasarse, 0 empate
"""

import random


# Valores de las cartas (J, Q, K valen 10; As vale 1 u 11)
CARD_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 10, 'Q': 10, 'K': 10, 'A': 11
}

DECK = list(CARD_VALUES.keys()) * 4  # Baraja estándar (52 cartas)


def draw_card():
    """Saca una carta aleatoria de la baraja infinita (con reemplazo)."""
    return random.choice(DECK)


def card_numeric_value(card):
    """Devuelve el valor numérico de la carta (As siempre se devuelve como 11 inicialmente)."""
    return CARD_VALUES[card]


def hand_value(hand):
    """
    Calcula el valor óptimo de una mano.
    
    Retorna:
        (total, usable_ace): total numérico y si hay un As contado como 11.
    """
    total = 0
    aces = 0

    for card in hand:
        val = CARD_VALUES[card]
        if card == 'A':
            aces += 1
        total += val

    # Reducir Ases de 11 a 1 si nos pasamos de 21
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    usable_ace = aces > 0 and (total - 10) <= 21 and total <= 21
    return total, usable_ace


def get_state(player_hand, dealer_visible_card):
    """
    Construye el estado del MDP.
    
    Retorna:
        (suma_jugador, valor_carta_dealer, as_utilizable)
        Ejemplo: (17, 6, False)
    """
    player_total, usable_ace = hand_value(player_hand)
    # La carta visible del dealer: As se representa como 1
    dealer_val = CARD_VALUES[dealer_visible_card]
    if dealer_visible_card == 'A':
        dealer_val = 1

    return (player_total, dealer_val, usable_ace)


class BlackjackEnv:
    """
    Entorno del Blackjack implementado como MDP tabular.
    
    Espacio de estados: 18 × 10 × 2 = 360 estados
    Espacio de acciones: {0: Stand, 1: Hit}
    """

    def __init__(self):
        self.player_hand = []
        self.dealer_hand = []
        self.done = False

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(self):
        """
        Inicia un nuevo episodio.
        
        El dealer reparte 2 cartas a cada uno; solo una carta del dealer es visible.
        Retorna el estado inicial.
        """
        self.player_hand = [draw_card(), draw_card()]
        self.dealer_hand = [draw_card(), draw_card()]
        self.done = False
        self._natural = False  # Flag de blackjack natural

        # Verificar blackjack natural del jugador (21 con 2 cartas)
        player_total, _ = hand_value(self.player_hand)
        if player_total == 21:
            self._natural = True  # Se resuelve en el primer step()

        state = get_state(self.player_hand, self.dealer_hand[0])
        return state

    # ─── Step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        """
        Ejecuta una acción.
        
        Args:
            action: 0 = PLANTARSE, 1 = PEDIR

        Retorna:
            (next_state, reward, done)
        """
        assert not self.done, "El episodio ya terminó. Llama reset() primero."

        # Blackjack natural: el jugador gana automáticamente (el agente
        # puede elegir cualquier acción — la mano se resuelve igual)
        if self._natural:
            return self._stand()

        if action == 1:  # PEDIR (Hit)
            return self._hit()
        else:            # PLANTARSE (Stand)
            return self._stand()

    def _hit(self):
        """El jugador pide una carta."""
        self.player_hand.append(draw_card())
        player_total, _ = hand_value(self.player_hand)

        if player_total > 21:
            # Jugador se pasó → pierde
            self.done = True
            next_state = get_state(self.player_hand, self.dealer_hand[0])
            return next_state, -1.0, True

        next_state = get_state(self.player_hand, self.dealer_hand[0])
        return next_state, 0.0, False

    def _stand(self):
        """El jugador se planta; el dealer juega su turno."""
        # El dealer pide cartas hasta llegar a 17 o más
        while True:
            dealer_total, _ = hand_value(self.dealer_hand)
            if dealer_total >= 17:
                break
            self.dealer_hand.append(draw_card())

        player_total, _ = hand_value(self.player_hand)
        dealer_total, _ = hand_value(self.dealer_hand)

        self.done = True
        next_state = get_state(self.player_hand, self.dealer_hand[0])

        # Determinar resultado
        if dealer_total > 21 or player_total > dealer_total:
            reward = 1.0   # Jugador gana
        elif player_total < dealer_total:
            reward = -1.0  # Jugador pierde
        else:
            reward = 0.0   # Empate

        return next_state, reward, True

    # ─── Información del entorno ──────────────────────────────────────────────

    @property
    def action_space(self):
        return [0, 1]  # 0=Stand, 1=Hit

    def render(self):
        """Muestra el estado actual del juego en consola."""
        p_total, p_ace = hand_value(self.player_hand)
        d_total, d_ace = hand_value(self.dealer_hand)
        print(f"  Jugador: {self.player_hand} → {p_total}"
              f"{'(As utilizable)' if p_ace else ''}")
        print(f"  Dealer:  {self.dealer_hand} → {d_total}"
              f"{'(As utilizable)' if d_ace else ''}")
