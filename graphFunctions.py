# Function to determine the line type string based on its type value
def get_line_type_string(line_type):
    return 'Underground' if line_type == 0 else 'Overhead'