########################################################################################################################
########## Sources
########################################################################################################################

# https://github.com/hkmztrk/DeepDTA/tree/master

########################################################################################################################
########## Import
########################################################################################################################

import torch
import logging
import numpy as np
from torch_geometric.data import Data
logger = logging.getLogger(__name__)

########################################################################################################################
########## Define dictionaries
########################################################################################################################

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


########################################################################################################################
########## Function
########################################################################################################################


def integer_label_string(sequence, tp):
    """
    Integer encoding for string sequence.
    Args:
        sequence (str): Drug or Protein string sequence.
        max_length: Maximum encoding length of input string.
    """
    if tp == 'drug':
        max_length = 150
        charset = CHARISOSMISET
    elif tp == 'protein':
        max_length = 1500
        charset = CHARPROTSET

    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            if tp == 'protein':
                letter = letter.upper()
            letter = str(letter)
            encoding[idx] = charset[letter]
        except KeyError:
            logger.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return Data(x=torch.from_numpy(encoding).to(torch.long).unsqueeze(dim=0))


if __name__ == '__main__':
    dr = 'COc1cc2c(cc1Cl)C(c1ccc(Cl)c(Cl)c1)=NCC2'
    pr = 'MSWSPSLTTQTCGAWEMKERLGTGGFGNVIRWHNQETGEQIAIKQCRQELSPRNRERWCLEIQIMRRLTHPNVVAARDVPEGMQNLAPNDLPLLAMEYCQGGDLRKYLNQFENCCGLREGAILTLLSDIASALRYLHENRIIHRDLKPENIVLQQGEQRLIHKIIDLGYAKELDQGSLCTSFVGTLQYLAPELLEQQKYTVTVDYWSFGTLAFECITGFRPFLPNWQPVQWHSKVRQKSEVDIVVSEDLNGTVKFSSSLPYPNNLNSVLAERLEKWLQLMLMWHPRQRGTDPTYGPNGCFKALDDILNLKLVHILNMVTGTIHTYPVTEDESLQSLKARIQQDTGIPEEDQELLQEAGLALIPDKPATQCISDGKLNEGHTLDMDLVFLFDNSKITYETQISPRPQPESVSCILQEPKRNLAFFQLRKVWGQVWHSIQTLKEDCNRLQQGQRAAMMNLLRNNSCLSKMKNSMASMSQQLKAKLDFFKTSIQIDLEKYSEQTEFGITSDKLLLAWREMEQAVELCGRENEVKLLVERMMALQTDIVDLQRSPMGRKQGGTLDDLEEQARELYRRLREKPRDQRTEGDSQEMVRLLLQAIQSFEKKVRVIYTQLSKTVVCKQKALELLPKVEEVVSLMNEDEKTVVRLQEKRQKELWNLLKIACSKVRGPVSGSPDSMNASRLSQPGQLMSQPSTASNSLPEPAKKSEELVAEAHNLCTLLENAIQDTVREQDQSFTALDWSWLQTEEEEHSCLEQAS'

    drug_seq = integer_label_string(dr, 'drug').x
    prot_seq = integer_label_string(pr, 'protein').x

    print(drug_seq, drug_seq.shape, drug_seq.dtype)
    print(prot_seq, prot_seq.shape, prot_seq.dtype)

    len(drug_seq)


    from torch import nn
    e = nn.Embedding(CHARISOSMILEN + 1, 128)
    c = nn.Conv1d(in_channels=150, out_channels=32, kernel_size=8)
    fc1_xt = nn.Linear(32 * 121, 128)
    print(e(drug_seq).shape)
    print(c(e(drug_seq)).shape)
    xt = c(e(drug_seq)).view(-1, 32 * 121)
    print(xt.shape)
    print(fc1_xt(xt).shape)

    e2 = nn.Embedding(CHARPROTLEN + 1, 128)
    c2 = nn.Conv1d(in_channels=1500, out_channels=32, kernel_size=8)
    fc2_xt = nn.Linear(32 * 121, 128)
    print(e2(prot_seq).shape)
    print(c2(e2(prot_seq)).shape)
    xt2 = c2(e2(prot_seq)).view(-1, 32 * 121)
    print(xt2.shape)
    print(fc2_xt(xt2).shape)



