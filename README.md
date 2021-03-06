# UROM
Python implementation of uplifted reduced order modeling (UROM) framework for the paper: 
### ["A long short-term memory embedding for hybrid uplifted reduced order models"](https://www.sciencedirect.com/science/article/pii/S0167278919307766) 
by Shady E. Ahmed, Omer San, Adil Rasheed, and Traian Iliescu.

Includes two test cases; 1D viscous Burgers problem, and 2D vortex merger problem. 

Each case has a a folder that contains field data generation, proper orthogonal decomposition (POD), Galerkin projection, training long short-term memory (LSTM) networks for UROM as well as fully nonintrusive reduced order model (NIROM), and testing of the three discussed frameworks, namely GROM, NIROM, and UROM.

Codes are implemented and tested using Python 3.7.4 and TensorFlow 2.0.0.

If you have any questions, comments and/or suggestions, please do not hesitate to contact me at: shady.ahmed@okstate.edu

