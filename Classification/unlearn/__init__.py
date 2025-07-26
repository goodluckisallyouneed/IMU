from .SSD import SSD
from .GA import GA,GA_l1
from .RL import RL
from .FT import FT,FT_l1
from .fisher import fisher,fisher_new
from .retrain import retrain
from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint
from .Wfisher import Wfisher
from .FT_prune import FT_prune
from .FT_prune_bi import FT_prune_bi
from .GA_prune_bi import GA_prune_bi
from .GA_prune import GA_prune
from .RL_pro import RL_proximal
from .boundary_ex import boundary_expanding
from .boundary_sh import boundary_shrink
from .SCAR import SCAR
from .SCAR import Random_l
from .NPO import NPO
from .IMU import IMU
from .SCAR_REID import SCAR_REID
from .SCAR_REID import Random_l_REID
from .IMU_REID import IMU_REID


def raw(data_loaders, model, criterion, args, mask=None):
    pass


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    elif name == "RL":
        return RL
    elif name == "GA":
        return GA
    elif name == "FT":
        return FT
    elif name == "FT_l1":
        return FT_l1
    elif name == "fisher":
        return fisher
    elif name == "retrain":
        return retrain
    elif name == "fisher_new":
        return fisher_new
    elif name == "wfisher":
        return Wfisher
    elif name == "FT_prune":
        return FT_prune
    elif name == "FT_prune_bi":
        return FT_prune_bi
    elif name == "GA_prune":
        return GA_prune
    elif name == "GA_prune_bi":
        return GA_prune_bi
    elif name == "GA_l1":
        return GA_l1
    elif name == "boundary_expanding":
        return boundary_expanding
    elif name == "boundary_shrink":
        return boundary_shrink
    elif name == "RL_proximal":
        return RL_proximal
    elif name == "SSD":
        return SSD
    elif name == "SCAR":
        return SCAR
    elif name == "Random_l":
        return Random_l
    elif name == "NPO":
        return NPO
    elif name == "IMU":
        return IMU
    elif name == "SCAR_REID":
        return SCAR_REID
    elif name == "Random_l_REID":
        return Random_l_REID
    elif name == "IMU_REID":
        return IMU_REID
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
