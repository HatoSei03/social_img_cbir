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

def load_plain_annotation(id, fpath):
	# jump to line id in csv
	line = read_line(id + 1, fpath)
	# Split the line by commas
	split_line = line.split(",")
	potential = split_line[1].split(";")
	relevant = split_line[2].split(";")
	
	# remove empty string and strip whitespace and newline characters
	potential = [x.strip() for x in potential if x.strip() != ""]
	relevant = [x.strip() for x in relevant if x.strip() != ""]
	
	return potential, relevant