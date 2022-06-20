# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from .bert_connector import Bert
# from .elmo_connector import Elmo
from .gpt_connector import GPT
from .transformerxl_connector import TransformerXL
from .ernie_connector import ERNIE

# from .roberta_connector import Roberta
from .luke_connector import Luke
from .ourroberta_connector import OurRoberta

try:
    # from .robertaent_connector import RobertaEnt
    from .robertaconcat_connector import RobertaConcat
    from .bertconcat_connector import BertConcat
except:
    print ('Miss Concat')

try:
    from .bart_connector import Bart
    from .bartconcat_connector import BartConcat
except:
    print ('Miss BART')

def build_model_by_name(lm, args, verbose=True):
    """Load a model by name and args.

    Note, args.lm is not used for model selection. args are only passed to the
    model's initializator.
    """
    MODEL_NAME_TO_CLASS = dict(
        # elmo=Elmo,
        # bert=Bert,
        # gpt=GPT,
        # transformerxl=TransformerXL,
        # roberta=Roberta,
        # luke=Luke,
        # ourroberta=OurRoberta,
        # robertaent=RobertaEnt,
        # robertaconcat=RobertaConcat,
        # bertconcat=BertConcat,
        # ernie=ERNIE,
        bart=Bart,
        bartconcat=BartConcat,
    )
    if lm not in MODEL_NAME_TO_CLASS:
        raise ValueError("Unrecognized Language Model: %s." % lm)
    if verbose:
        print("Loading %s model..." % lm)
    return MODEL_NAME_TO_CLASS[lm](args)
