# Decision_Tree_with_Adaboost
Implentation of the ID3 Decision Tree algorithm with Adaboost

-- See writeup.pdf to see implementation details and for usage information --

• Usage: To run this code, run main.py with cmd-line arguments:

  ▪ main <training_data> <hypothesis_out><learning_type><test><labels>
  
  ▪ <training_data> is the filename for the training data I include in the zip (trainingData.txt)
	
  ▪ <hypothesis_out> is the filename to which the serialized model will be written
	
  ▪ <learning_type> is either ‘ada’ or ‘dt’ for adaboost or decision tree
	
  ▪ <test> is the filename for test data you wish to use (here is testData.txt)
	
  ▪ <labels> is an OPTIONAL argument if you wish to provide a new-line separated text file with the correct labels for test data.       If labels are provided, then evaluate() will be called in main.py and will print the accuracy (percentage) 
    of the predictions with the given model.
