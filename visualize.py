import os
import json
import matplotlib.pyplot as plt

from configs import PATH_TO_EXPERIMENTS

# Visualize history
# Plot history: Loss
# plt.plot(history.history['val_loss'])
# plt.title('Validation loss history')
# plt.ylabel('Loss value')
# plt.xlabel('No. epoch')
# plt.show()
#
# # Plot history: Accuracy
# print(history.history)
# plt.plot(history.history['val_acc'])
# plt.title('Validation accuracy history')
# plt.ylabel('Accuracy value (%)')
# plt.xlabel('No. epoch')
# plt.show()


def read_single_exp(path_to_exp, target_keys=('acc', 'val_acc')):
	with open(path_to_exp, 'r') as json_file:
		data = json.load(json_file)

	train_metric = data[target_keys[0]]
	val_metric = data[target_keys[1]]

	train_metric = list(train_metric.values())
	val_metric = list(val_metric.values())
	return train_metric, val_metric


def comparison_viz():
	pass


def train_val_viz():
	pass


if __name__ == '__main__':
	train_colors = ['red', 'green']
	val_colors = ['blue', 'orange']

	# Experiment config
	experiments_to_visualize = ['e_0001']
	plt_title = 'Validation accuracy history'
	plt_x_label = 'No. epoch'
	plt_y_label = 'Accuracy value (%)'

	exps_train_data, exps_val_data = [], []
	for exp_name in experiments_to_visualize:
		train_metric, val_metric = read_single_exp(os.path.join(PATH_TO_EXPERIMENTS, "{}.json".format(exp_name)))
		exps_train_data.append(train_metric)
		exps_val_data.append(val_metric)

	for i, (train_line, val_line) in enumerate(zip(exps_train_data, exps_val_data)):
		plt.plot(train_line, color=train_colors[i], label=experiments_to_visualize[i])
		plt.plot(val_line, color=val_colors[i], label=experiments_to_visualize[i])
	plt.title(plt_title)
	plt.xlabel(plt_x_label)
	plt.ylabel(plt_y_label)
	plt.legend()
	plt.show()
