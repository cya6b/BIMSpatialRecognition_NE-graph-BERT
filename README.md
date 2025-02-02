# BIMSpatialRecognition_NE-graph-BERT
* Code and instructions for our paper: SCIE Journal Under Review (2025-January)
# Introduction
The Spacegraph dataset was created using 307 two-dimensional (2D) floor plans representing three building types: schools, apartments, and office buildings. We gathered the 2D floor plan images through the Google search engine. A custom tagging system was developed to semantically annotate the functional spaces and their relationships directly on the 2D floor plan images. Subsequently, we developed the NE-Graph-BERT model, which leverages edge features to enhance node features and can be employed to automatically identify different types of functional spaces.

# SpaceGraph dataset development and annotation
Customized semantic annotation tools

https://github.com/user-attachments/assets/44a97946-9f7e-423f-a4c3-c7b832fa11fc
 
 
# Catalog Structure Description
    ├── ReadMe.md           // help file
    
    ├── dataset    // This section contains SpaceGraph's final JSON format files, along with the original 2D flat image files.

    ├── code              // Core Code Documentation

    │   ├── MethodWLENodeColoring.py (for WLE code computing)

    │   ├── MethodGraphBatching.py (for subgraph batching)

    │   ├── MethodHopDistance.py (for hop distance computing)
    
    ├── result             // Contains subgraph batch results, WLE results, Hop embedding results, and pre-trained models.

    ├── script_1_preprocess.py             // Compute node WLE code, intimacy based subgraph batch, node hop distance.

    ├── script_2_pre_train.py             // For pre-training the NE-Graph-BERT.

    ├── script_3_fine_tuning.py             // As the entry point to run the model on node classification.

    └── script_4_evaluation_plots.py             // Plots drawing and results evaluation purposes.
 
# Contact
If you have any question, feel free to contact me at cya187508866962021@163.com

Fujian Normal University School of Geographical Sciences

# Prerequisites
* python==3.8
* pytorch

  [PyTorch](https://pytorch.org/get-started/locally/)
* numpy
* sklearn
* argparse
* pytorch geometric

  [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
* networkx
* pandas
 
# Execution Guide
###### In order to replicate the results mentioned in paper, please follow the following steps:
Run the command "python script_1_preprocess.py" to compute node WLE code, intimacy based subgraph batch, node hop distance.

    python script_1_preprocess.py

Run the command "python script_3_fine_tuning.py" as the entry point to run the model on BIM spatial recognition.

    python script_3_fine_tuning.py

Run the command "python script_4_evaluation_plots.py" for plots drawing and results evaluation purposes.

    python script_4_evaluation_plots.py

# demo
* The demo link is located at the top right corner of the interface.

Users simply need to enter the name of the desired functional space name in the search box, such as:

    bedroom
or    

    bathroom
    
or    

    living room 

the demo will automatically identify the corresponding functional space in the model and highlight it.


# Citation
    @onproceeding{
	    author={Chen, Y. and Jiang, H.},
	    doi={},
	    journal={},
	    year={2025},
    }

 
 
