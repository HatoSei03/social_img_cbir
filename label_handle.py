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
			lines.append(decoded_line)
	return lines