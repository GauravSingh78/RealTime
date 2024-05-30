# import os
# import pickle
# import numpy as np
# from sklearn.tree import _tree
# from sklearn.ensemble import RandomForestClassifier

# def extract_and_save_rules(pickle_file_path, output_folder):
#     # Extract file name from path
#     pickle_file_name = os.path.basename(pickle_file_path)
    
#     # Load the model
#     with open(pickle_file_path, 'rb') as file:
#         model_data = pickle.load(file)

#     loaded_model = model_data['model']
#     feature_names = model_data['feature_names']
#     class_names = None

    
#     def get_rules(tree, feature_names, class_names):
#         try:
#             tree_ = tree.tree_
#             feature_name = [
#                 feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
#                 for i in tree_.feature
#             ]

#             paths = []
#             path = []

#             def recurse(node, path, paths):
#                 if tree_.feature[node] != _tree.TREE_UNDEFINED:
#                     name = feature_name[node]
#                     threshold = tree_.threshold[node]
#                     p1, p2 = list(path), list(path)
#                     p1 += [f"({name} <= {np.round(threshold, 20)})"]
#                     recurse(tree_.children_left[node], p1, paths)
#                     p2 += [f"({name} > {np.round(threshold, 20)})"]
#                     recurse(tree_.children_right[node], p2, paths)
#                 else:
#                     path += [(tree_.value[node], tree_.n_node_samples[node])]
#                     paths += [path]

#             recurse(0, path, paths)

#             samples_count = [p[-1][1] for p in paths]
#             ii = list(np.argsort(samples_count))
#             paths = [paths[i] for i in reversed(ii)]

#             rules = []
#             for path in paths:
#                 rule = "if "

#                 for p in path[:-1]:
#                     if rule != "if ":
#                         rule += " and "
#                     rule += str(p)

#                 if class_names is None:
#                     rule += ": return " + str(np.round(path[-1][0][0][0], 20))
#                 else:
#                     classes = path[-1][0][0]
#                     l = np.argmax(classes)
#                     rule += f"class: {class_names[l]} (proba: {np.round(10000000.0 * classes[l] / np.sum(classes), 20)}%)"

#                 rules += [rule]

#             return rules
#         except (AttributeError, NameError) as e:
#             print(f"Error extracting rules: {e}")
#             return None

    
#     def get_all_rules(forest, feature_names, class_names):
#         all_rules = []
#         for tree in forest.estimators_:
#             rules = get_rules(tree, feature_names, class_names)
#             if rules:
#                 all_rules.append(rules)
#         return all_rules

    
#     all_rules = get_all_rules(loaded_model, feature_names, class_names)

    
#     output_file_name = os.path.splitext(pickle_file_name)[0] + '_rules.txt'
#     output_path = os.path.join(output_folder, output_file_name)
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     with open(output_path, 'w') as file:
#         for i, tree_rules in enumerate(all_rules):
#             file.write(f"\nRules from Tree {i+1}\n")
#             for rule in tree_rules:
#                 file.write(rule + '\n')

#     print(f"Rules have been extracted and written to {output_path}")


# if __name__ == '__main__':
#     pickle_file_path = r"C:\Users\ASUS\OneDrive\Desktop\EndSem_Project\trained_model\Random_Forest_20.pkl"
#     output_folder = 'RulesFolder'
#     extract_and_save_rules(pickle_file_path, output_folder)


import os
import pickle
import numpy as np
from sklearn.tree import _tree
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import filedialog, simpledialog

def extract_and_save_rules(pickle_file_path):
    # Create a Tkinter window
    root = tk.Tk()
    # root.withdraw()  # Hide the root window

    # Ask user to choose the file path for saving
    save_path = filedialog.asksaveasfilename(defaultextension=".txt", title="Save Rules As", filetypes=[("Text files", "*.txt")])
    
    if not save_path:
        print("No file saved. Exiting.")
        return

    output_folder = os.path.dirname(save_path)
    filename = os.path.basename(save_path)

    # root.mainloop()
    root.destroy()
    # Extract file name from path
    pickle_file_name = os.path.basename(pickle_file_path)
    
    # Load the model
    with open(pickle_file_path, 'rb') as file:
        model_data = pickle.load(file)

    loaded_model = model_data['model']
    feature_names = model_data['feature_names']
    class_names = None

    
    def get_rules(tree, feature_names, class_names):
        try:
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]

            paths = []
            path = []

            def recurse(node, path, paths):
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    p1, p2 = list(path), list(path)
                    p1 += [f"({name} <= {np.round(threshold, 20)})"]
                    recurse(tree_.children_left[node], p1, paths)
                    p2 += [f"({name} > {np.round(threshold, 20)})"]
                    recurse(tree_.children_right[node], p2, paths)
                else:
                    path += [(tree_.value[node], tree_.n_node_samples[node])]
                    paths += [path]

            recurse(0, path, paths)

            samples_count = [p[-1][1] for p in paths]
            ii = list(np.argsort(samples_count))
            paths = [paths[i] for i in reversed(ii)]

            rules = []
            for path in paths:
                rule = "if "

                for p in path[:-1]:
                    if rule != "if ":
                        rule += " and "
                    rule += str(p)

                if class_names is None:
                    rule += ": return " + str(np.round(path[-1][0][0][0], 20))
                else:
                    classes = path[-1][0][0]
                    l = np.argmax(classes)
                    rule += f"class: {class_names[l]} (proba: {np.round(10000000.0 * classes[l] / np.sum(classes), 20)}%)"

                rules += [rule]

            return rules
        except (AttributeError, NameError) as e:
            print(f"Error extracting rules: {e}")
            return None

    
    def get_all_rules(forest, feature_names, class_names):
        all_rules = []
        for tree in forest.estimators_:
            rules = get_rules(tree, feature_names, class_names)
            if rules:
                all_rules.append(rules)
        return all_rules

    
    all_rules = get_all_rules(loaded_model, feature_names, class_names)

    with open(save_path, 'w') as file:
        for i, tree_rules in enumerate(all_rules):
            file.write(f"\nRules from Tree {i+1}\n")
            for rule in tree_rules:
                file.write(rule + '\n')

    print(f"Rules have been extracted and saved to {save_path}")



if __name__ == '__main__':
    pickle_file_path = r"C:\Users\ASUS\OneDrive\Desktop\EndSem_Project\trained_model\Random_Forest_20.pkl"
    extract_and_save_rules(pickle_file_path)
