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


def read_single_exp(path_to_exp, target_key='val_acc'):
	with open(path_to_exp, 'r') as json_file:
		data = json.load(json_file)

	data = data[target_key]
	target_data = list(data.values())
	return target_data



if __name__ == '__main__':
	colors = ['red', 'green', 'blue', 'orange']

	# Experiment config
	experiments_to_visualize = ['e_0001']
	plt_title = 'Validation accuracy history'
	plt_x_label = 'No. epoch'
	plt_y_label = 'Accuracy value (%)'

	exps_data = []
	for exp_name in experiments_to_visualize:
		exp_data = read_single_exp(os.path.join(PATH_TO_EXPERIMENTS, "{}.json".format(exp_name)))
		exps_data.append(exp_data)

	for i, line in enumerate(exps_data):
		plt.plot(line, color=colors[i], label=experiments_to_visualize[i])
	plt.title(plt_title)
	plt.xlabel(plt_x_label)
	plt.ylabel(plt_y_label)
	plt.legend()
	plt.show()
