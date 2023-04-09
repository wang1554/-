import pickle
import matplotlib.pyplot as plt
if __name__ == '__main__':
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    for key in model.params.keys():
        if "w" in key:
            weights = model.params[key]
            plt.imshow(weights, cmap='coolwarm')
            plt.title(f"Weight heatmap of {key}")
            plt.colorbar()
            plt.show()





