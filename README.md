# Neuromorphic
TensorFlow code for analysis of Neuromorphic Circuits

Classes for constructing MLP and Convolutional layers whose behavior mimics that of actual
neuromorphic hardware. Additionally, algorithms for training in a HW-aware manner are included.

Initial checkin contains clases for layers which contain dedicated positive and negative weights.
This greatly simplifies the HW and reduces circuit area. No appreciable loss of accuracy
is incurred when training with the appropriate HW-aware training algorithm.
