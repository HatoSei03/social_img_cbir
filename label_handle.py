from annotated_text import annotation


def read_upload(file):
    lines = []
    for line in file:
        # Decode the line from bytes to string and strip any leading/trailing whitespace
        decoded_line = line.decode("utf-8").strip()
        # Append the line to the array
        lines.append(decoded_line)
    
    return lines

def load_label_from_path(path):
    lines = []
    with open(path, 'r', encoding="utf-8") as f:
        # Read each line in the text file
        for line in f:
            # Strip any leading/trailing whitespace
            decoded_line = line.strip()
            # Append the line to the array
            lines.append((decoded_line, ""))
    return lines


def read_line(line_number, fpath):
    with open(fpath, mode='rb') as file:
        for _ in range(line_number-1):
            file.readline()  # Đọc và bỏ qua dòng
        line = file.readline().decode('utf-8')  # Đọc dòng thứ 1000 và giải mã
    return line

def load_annotation(id, fpath):
    # jump to line id in csv
    line = read_line(id + 1, fpath)
    # Split the line by commas
    split_line = line.split(",")
    potential = split_line[1].split(";")
    potential_color = split_line[2].split(";")
    relevant = split_line[3].split(";")
    relevant_color = split_line[4].split(";")
    
    # tuple potential with color
    po_ano = []
    for i in range(len(potential)):
        if potential[i] == "":
            continue
        annou = annotation(
            potential[i], "", background=potential_color[i], color="black")
        po_ano.append(annou)
    
    # tuple relevant with color
    re_ano = []
    for i in range(len(relevant)):
        if relevant[i] == "":
            continue
        annou = annotation(
            relevant[i], "", background=relevant_color[i], color="#000000")
        re_ano.append(annou)
    
    return po_ano, re_ano

def load_plain_annotation(id, fpath):
	# jump to line id in csv
	line = read_line(id + 1, fpath)
	# Split the line by commas
	split_line = line.split(",")
	potential = split_line[1].split(";")
	relevant = split_line[3].split(";")
	
	return potential, relevant