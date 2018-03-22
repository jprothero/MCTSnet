EMB_DIMS = 128
EMB_IN = (2, 6, 7) #player and opponent boards overlapped: 2 * 6 * 7 
EMB_FILTERS = 64

POL_IN = EMB_DIMS#embedding, so it will match embedding dims
POL_LEN = 42 #6*7 connect4 board

READOUT_IN = EMB_DIMS

BACKUP_IN = EMB_DIMS*2+1+POL_LEN #inputs are embedding_dims*2 + 1(reward) + action(1 hot policy?)
 