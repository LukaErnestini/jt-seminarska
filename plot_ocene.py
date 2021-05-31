import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

files = glob.glob('oceneHaiku/*.txt')

results = []
human_author = [True, False, False, False, True, True, False, True, False, True,
                False, False, True, False, True, False, False, False, True, False]

human_indice = [i for i in range(len(human_author)) if human_author[i]]
ai_indice = [i for i in range(len(human_author)) if not human_author[i]]

for file in files:
    df = pd.read_csv(file, index_col=None, header=None)
    results.append(df.to_numpy())

results = np.array(results)

# xs_ai = np.zeros(len(ai_indice))
# xs_h = np.zeros(len(human_indice))
# ys_ai = np.zeros(len(ai_indice))
# ys_h = np.zeros(len(human_indice))

xs_ai = []
xs_h = []
ys_ai = []
ys_h = []
for person in results:
    x_ai = person[:,0][ai_indice]
    y_ai = person[:,1][ai_indice]
    x_h = person[:,0][human_indice]
    y_h = person[:,1][human_indice]

    # plt.scatter(x_ai, y_ai, c='r', label='AI-generated')
    # plt.scatter(x_h, y_h, c='g', label='human author')
    # plt.xlabel('How much do you like the poem [1-5]')
    # plt.ylabel('How much do you believe that the author is human [1-5]')
    # plt.legend()
    # plt.show()

    # xs_ai += x_ai / len(results)
    # xs_h += x_h / len(results)
    # ys_ai += y_ai / len(results)
    # ys_h += y_h / len(results)

    xs_ai.append(x_ai)
    xs_h.append(x_h)
    ys_ai.append(y_ai)
    ys_h.append(y_h)

xs_ai = np.array(xs_ai)
xs_h = np.array(xs_h)
ys_ai = np.array(ys_ai)
ys_h = np.array(ys_h)

x_mean_ai = np.mean(xs_ai, axis=0)
x_mean_h = np.mean(xs_h, axis=0)
y_mean_ai = np.mean(ys_ai, axis=0)
y_mean_h = np.mean(ys_h, axis=0)



fig, ax = plt.subplots()
ax.scatter(x_mean_ai, y_mean_ai, c='r', label='AI-generated')
ax.scatter(x_mean_h, y_mean_h, c='g', label='human author')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
for i in range(len(ai_indice)):
    ax.annotate(ai_indice[i] + 1, (x_mean_ai[i], y_mean_ai[i]))
for i in range(len(human_indice)):
    ax.annotate(human_indice[i] + 1, (x_mean_h[i], y_mean_h[i]))
plt.xlabel('How much do you like the poem [1-5]')
plt.ylabel('How strong do you believe that the author is human [1-5]')
plt.legend()
plt.savefig('average_score_poems_2.pdf')