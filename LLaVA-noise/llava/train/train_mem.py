import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from llava.train.train import train

if __name__ == "__main__":
    import os
    import wandb   
    # os.environ["WANDB_MODE"]="offline"
    train(attn_implementation="flash_attention_2")