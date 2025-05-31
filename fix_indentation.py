"""Script to fix indentation issues in streamlit_dashboard_simple.py"""

def fix_file():
    with open('streamlit_dashboard_simple.py', 'r') as file:
        lines = file.readlines()
    
    fixed_lines = []
    for i, line in enumerate(lines):
        # Fix specific indentation issues we identified
        if "category_count = len(" in line and "palette" in line:
            # Split the line at palette
            parts = line.split("palette")
            fixed_lines.append(f"{parts[0]}\n")
            fixed_lines.append(f"    palette{parts[1]}")
        elif "  bars = sns.barplot" in line:
            # Fix the indentation of bar lines
            fixed_lines.append(line.replace("  bars", "    bars"))
        else:
            fixed_lines.append(line)
    
    with open('streamlit_dashboard_simple.py', 'w') as file:
        file.writelines(fixed_lines)
    
    print("Fixed indentation issues in streamlit_dashboard_simple.py")

if __name__ == "__main__":
    fix_file()
