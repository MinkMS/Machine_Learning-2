input_file = r"C:\Users\Mink\Documents\GitHub\Dataset-Save-Place\Air Quality\AirQualityUCI.csv"
output_file = r"C:\Users\Mink\Documents\GitHub\Dataset-Save-Place\Air Quality\AirQualityUCI_cleaned.csv"

with open(input_file, "r", encoding="utf-8") as fin:
    lines = fin.readlines()

cleaned_lines = []
for line in lines:
    # Remove trailing separators and whitespace
    line = line.rstrip()
    if line.endswith(",,"):
        line = line[:-2]
    cleaned_lines.append(line + "\n")

with open(output_file, "w", encoding="utf-8") as fout:
    fout.writelines(cleaned_lines)

print("Cleaning done!")
print(f"Clean file saved as: {output_file}")
