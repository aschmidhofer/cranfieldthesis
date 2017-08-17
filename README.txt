User guide on how to use the software provided.

Prerequisites:
- python3 (https://www.python.org/downloads/release/python-362/)
- TensorFlow (https://www.tensorflow.org/install/)


To preform a retrain execute the 'retrain.py' script
with the following arguments:
--bottleneck_dir directory where to cache the bottlenecks
--how_many_training_steps number of training steps (or -1 for using dynamic steps with a stopping criterium)
--model_dir where to store the Inception model
--output_graph where to stored the retrained network
--output_labels where to store the labels which correspond to the output nodes (those are the class names, but the order might change from one execution to another)
--image_dir where the dataset containing images is located (this folder must contain the 3 subfolders "training", "validation" and "testing", and each of the subfolders must have subfolders that correspond to the classes, which should then contain the according images)
--learning_rate the initial learning rate to use
--output_dir directory to store statistics 
--bottlenecks_exist add this argument to skip the bottleneck creation step (the program will fail if bottlenecks were not created in the folder specified, bottlenecks that are already created will be skipped even if this argument is not provided, it just saves a few seconds of going through all of them) 

Example:
python3 retrain.py --bottleneck_dir=/home/andy/output/food11/bottlenecks --how_many_training_steps -1 --eval_step_interval=1 --model_dir=/home/andy/inception --output_graph=/home/andy/output/food11/test.pb --output_labels=/home/andy/output/food11/labels.txt --image_dir=/home/andy/imgs/Food-11 --learning_rate=0.01 --output_dir=./mystats


To create plots use the 'visualize.py' script.
either provide a directory that contain the statistics that were created by retraining

python3 visualize.py --file=./mystats/adjLR_vL_0p15_dynamic.data

or by providing a directory that contains statistic files

python3 visualize.py --dir=./mystats/


The 'confusionmatrix.py' script works in the same way with the additional parameter 
--labels that should point to a text file containing the corresponding class names

such a file called 'hrlabels.txt' is provided in the dataset directory.

Example:
python3 confusionmatrix.py --file=./mystats/adjLR_vL_0p15_dynamic.data --labels=../dataset/hrlabels.txt 


For queries please contact the author at
a.schmidhofer [at] cranfield [dot] ac [dot] uk

