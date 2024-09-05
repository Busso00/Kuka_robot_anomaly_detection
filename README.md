# MLinApps-Proj-FP01
The project involves the use of the time series dataset coming from an industrial robot named Kuka; the challenge is detecting the robots malfunctions and anomalies using machine learning models trained using these data. This analysis proves an interesting occasion to test various SOTA models and their performance in this application.


NOTE: file collision must be changed into an unique collision sheet


# Brief description of project structure
main.py

for personal testing purpose, in the main the opt['experiment'] is set manually to the model name by who is testing.
the principal flow of this script is to prepare the data by using different loader imported from load_data.py, preset the experiment hyperparameter in the experiment class in the /experiment folder,
these job are done by the function setup_experiment.




