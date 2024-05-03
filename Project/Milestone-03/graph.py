import matplotlib.pyplot as plt

# Data
values = [73, 68, 69, 70, 73, 67]
labels = ["S1", "S2", "S3", "S4", "S5", "S6"]

legend_labels = {
    "S1": "Decision Tree Misc",
    "S2": "Simple Perceptron TFIDF",
    "S3": "SVM Bag of Words",
    "S4": "SVM TFIDF",
    "S5": "Majority Ensemble of Mix",
    "S6": "Neural Network Glove",
}

# Define light colors
colors = ["#FFD700", "#ADD8E6", "#90EE90", "#FFA07A", "#D3D3D3", "#FFC0CB"]

fig = plt.figure()
bars = plt.bar(labels, values, color=colors)
plt.bar(labels, values, color=colors)
plt.xlabel("Submission")
plt.ylabel("Accuracy %")
plt.title("Kaggle Submissions")

for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(values[i]) + "%", ha="center", va="bottom")


# Add legend to each bar
for bar, label in zip(bars, labels):
    print(label)
    bar.set_label(legend_labels[label])

# Set y-axis lower limit to 50
plt.ylim(50, 100)

plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
# plt.show()

# Save the figure
fig.savefig("kaggle-accuracy.png")
