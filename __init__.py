# from .mine_nodes import *
from .CXH_PhotoMaker import *
from .CXH_PhotoMaker_Batch import *

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CXH_PhotoMaker":CXH_PhotoMaker,
    "CXH_PhotoMaker_Batch":CXH_PhotoMaker_Batch
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CXH_PhotoMaker": "CXH_PhotoMaker",
    "CXH_PhotoMaker_Batch":"CXH_PhotoMaker_Batch"
}
