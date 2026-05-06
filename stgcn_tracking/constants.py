N_HIS      = 50   # 5 segundos de historia  @ 10fps
N_PRED     = 30   # 3 segundos a predecir   @ 10fps
N_FRAME    = N_HIS + N_PRED  # 80 frames por secuencia
STRIDE     = 5    # paso de la ventana deslizante (0.5 segundos)
N_PLAYERS  = 22   # jugadores en el campo a la vez (11 por equipo)
N_NODES    = N_PLAYERS + 1  # + balón = 23
N_FEAT     = 2    # (x, y)
MAX_NAN_RATIO = 0.20

import os as _os
OUTPUT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'stgcn_data')