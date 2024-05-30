# import os

# def generate_verilog(rules, tree_number):
#     module_name = f"RF_model_{tree_number}"
#     verilog_code = f"module {module_name}(vinp, vdd, pd, out);\n\tinput wreal vinp;\n\tinput wreal vdd;\n\tinput wreal pd;\n\tinput wreal out;\n\tparameter real temp=15;\n\treal prediction;\n\nalways @(*) begin\n"

#     for i, rule in enumerate(rules):
#         condition = rule.replace("return", "").strip()
#         condition = condition.replace("and", "&&") 
#         condition = condition.replace("temperature", "temp") 
#         condition = condition.replace(";", "")  
#         value = rules[rule]

#         if i == 0:
#             verilog_code += f"    if ({condition}) begin\n"
#         else:
#             verilog_code += f"    else if ({condition}) begin\n"

#         verilog_code += f"        prediction = {value};\n"
#         verilog_code += "    end\n"

#     verilog_code += "    else begin\n"
#     verilog_code += "        prediction = 0.0;\n"
#     verilog_code += "    end\n"
#     verilog_code += "end\n"
#     verilog_code += "assign out = prediction;\n"
#     verilog_code +="endmodule\n\n"

#     return verilog_code

# def read_rules_from_file(file_path):
#     rules = {}
#     with open(file_path, "r") as file:
#         lines = file.readlines()

#     current_tree_number = None
#     current_rules = {}

#     for line in lines:
#         if "Rules from Tree" in line:
#             if current_tree_number is not None:
#                 rules[current_tree_number] = current_rules
#             current_tree_number = int(line.split()[-1])
#             current_rules = {}
#         elif line.startswith("if"):
#             parts = line.split(":")
#             condition = parts[0][2:].strip()
#             value = parts[1].strip().split()[1]
#             current_rules[condition.strip()] = value.strip()

#     if current_tree_number is not None:
#         rules[current_tree_number] = current_rules

#     return rules

# def generate_and_save_verilog(rules_file, output_folder):
#     all_rules = read_rules_from_file(rules_file)

#     verilog_code = ""
#     for tree_number, rules in all_rules.items():
#         verilog_code += generate_verilog(rules, tree_number)

#     print(rules_file)
#     file_name = os.path.splitext(os.path.basename(rules_file))[0]
#     output_file_name = file_name + '_verilog.txt'
#     output_file_path = os.path.join(output_folder, output_file_name)
#     with open(output_file_path, "w") as file:
#         file.write(verilog_code)

#     print("Verilog modules generated and saved to", output_file_path)

# if __name__ == "__main__":
#     rules_file_path = input("Enter the path to the rules file: ")
#     output_folder_path = input("Enter the path to the output folder: ")
#     generate_and_save_verilog(rules_file_path, output_folder_path)





import os
import tkinter as tk
from tkinter import filedialog

def generate_verilog(rules, tree_number):
    module_name = f"RF_model_{tree_number}"
    verilog_code = f"module {module_name}(vinp, vdd, pd, out);\n\tinput wreal vinp;\n\tinput wreal vdd;\n\tinput wreal pd;\n\tinput wreal out;\n\tparameter real temp=15;\n\treal prediction;\n\nalways @(*) begin\n"

    for i, rule in enumerate(rules):
        condition = rule.replace("return", "").strip()
        condition = condition.replace("and", "&&") 
        condition = condition.replace("temperature", "temp") 
        condition = condition.replace(";", "")  
        value = rules[rule]

        if i == 0:
            verilog_code += f"    if ({condition}) begin\n"
        else:
            verilog_code += f"    else if ({condition}) begin\n"

        verilog_code += f"        prediction = {value};\n"
        verilog_code += "    end\n"

    verilog_code += "    else begin\n"
    verilog_code += "        prediction = 0.0;\n"
    verilog_code += "    end\n"
    verilog_code += "end\n"
    verilog_code += "assign out = prediction;\n"
    verilog_code +="endmodule\n\n"

    return verilog_code

def read_rules_from_file(file_path):
    rules = {}
    with open(file_path, "r") as file:
        lines = file.readlines()

    current_tree_number = None
    current_rules = {}

    for line in lines:
        if "Rules from Tree" in line:
            if current_tree_number is not None:
                rules[current_tree_number] = current_rules
            current_tree_number = int(line.split()[-1])
            current_rules = {}
        elif line.startswith("if"):
            parts = line.split(":")
            condition = parts[0][2:].strip()
            value = parts[1].strip().split()[1]
            current_rules[condition.strip()] = value.strip()

    if current_tree_number is not None:
        rules[current_tree_number] = current_rules

    return rules

def generate_and_save_verilog(rules_file, output_folder):
    all_rules = read_rules_from_file(rules_file)

    verilog_code = ""
    for tree_number, rules in all_rules.items():
        verilog_code += generate_verilog(rules, tree_number)

    file_name = os.path.splitext(os.path.basename(rules_file))[0]
    output_file_name = file_name + '_verilog.txt'
    output_file_path = os.path.join(output_folder, output_file_name)
    with open(output_file_path, "w") as file:
        file.write(verilog_code)

    print("Verilog modules generated and saved to", output_file_path)

def ask_folder_and_filename():
    # Create a Tkinter window
    
    root = tk.Tk()
    # root.withdraw()  # Hide the root window

    # Ask user to choose the file path for saving
    save_path = filedialog.asksaveasfilename(defaultextension=".txt", title="Save Verilog Module As", filetypes=[("Text files", "*.txt")])
    if not save_path:
        print("No file saved. Exiting.")
        return None, None
    output_folder = os.path.dirname(save_path)
    filename = os.path.basename(save_path)

    # root.mainloop()
    root.destroy()
    return output_folder, filename

if __name__ == "__main__":
    # Provide the path to the rules file
    rules_file_path = r"C:\Users\ASUS\OneDrive\Desktop\django\Region.txt"

    output_folder, _ = ask_folder_and_filename()
    if output_folder:
        generate_and_save_verilog(rules_file_path, output_folder)
