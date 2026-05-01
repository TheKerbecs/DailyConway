import json
import sys
import os

def grid_to_rle(grid):
    rle_parts = []
    for row in grid:
        count = 0
        current_char = row[0]
        row_rle = ""
        for char in row:
            if char == current_char:
                count += 1
            else:
                c = 'b' if current_char == '0' else 'o'
                row_rle += (str(count) if count > 1 else "") + c
                current_char = char
                count = 1
                
        # Add the last run, but omit trailing dead cells ('b') 
        # as per the standard RLE specification
        if current_char == '1': 
            row_rle += (str(count) if count > 1 else "") + 'o'
            
        rle_parts.append(row_rle)
    
    # Join rows with $ and terminate with !
    rle_str = "$".join(rle_parts) + "!"
    
    # Standard RLE limits lines to 70 characters
    lines = []
    for i in range(0, len(rle_str), 70):
        lines.append(rle_str[i:i+70])
        
    return "\n".join(lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python json_to_rle.py <path_to_json_or_folder>")
        sys.exit(1)
        
    path = sys.argv[1]
    
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json")]
    else:
        print("Invalid path")
        sys.exit(1)
        
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            grid = data["start_grid"]
            x = len(grid[0])
            y = len(grid)
            
            header = f"x = {x}, y = {y}, rule = B3/S23:T{x},{y}"
            rle_body = grid_to_rle(grid)
            
            print(f"--- {os.path.basename(fpath)} ---")
            print(header)
            print(rle_body)
            print()
            
        except Exception as e:
            print(f"Failed to process {fpath}: {e}")

if __name__ == "__main__":
    main()
