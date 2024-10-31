import struct
import sys

def read_binary_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            binary_data = file.read(2)  # Read 2 bytes for an int16 value
            number = struct.unpack('h', binary_data)[0]
            print(f"The number in the binary file is: {number}")
    except FileNotFoundError:
        print("Error: File not found.")
    except struct.error:
        print("Error: Could not unpack the data. Ensure the file contains a 2-byte integer.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_binary.py <binary_file_path>")
    else:
        file_path = sys.argv[1]
        read_binary_file(file_path)
