target_indices = get_target_indices_by_name(fg, "artists", "The")
print("Indices for :", target_indices)



target_indices = get_target_indices_by_name(fg, "artists", "Frank")
print("Indices for :", target_indices)


def get_target_indices_by_name(df, name_column, target_name):

    def match_name(cell):
        """
        Check if the target name exists in the cell, which may be a string, list, or composite string.
        """
        try:
            # Attempt to parse lists or composite strings
            parsed = ast.literal_eval(cell) if isinstance(cell, str) else cell
        except (ValueError, SyntaxError):
            # If parsing fails, treat it as a simple string
            parsed = cell
        
        # Handle lists or composite values
        if isinstance(parsed, list):
            return target_name in parsed
        elif isinstance(parsed, str):
            # Check for partial or exact match using regex
            return bool(re.search(rf'\b{re.escape(target_name)}\b', parsed, flags=re.IGNORECASE))
        return False
    
    # Apply the match function and get indices
    matching_indices = df[df[name_column].apply(match_name)].index.tolist()
    return matching_indices



import ast
import re
def get_target_indices_by_name(df, name_column, target_name):
    def match_name(cell):
        try:
            # Attempt to parse lists or composite strings
            parsed = ast.literal_eval(cell) if isinstance(cell, str) else cell
        except (ValueError, SyntaxError):
                # If parsing fails, treat it as a simple string
                parsed = cell
            
            # Handle lists or composite values
        if isinstance(parsed, list):
                return target_name in parsed
        elif isinstance(parsed, str):
                # Check for partial or exact match using regex
            return bool(re.search(rf'\b{re.escape(target_name)}\b', parsed, flags=re.IGNORECASE))
        return False
    
    matching_indices = df[df[name_column].apply(match_name)].index.tolist()
    return matching_indices