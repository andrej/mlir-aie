#!/usr/bin/env python3
"""
Script to add missing tile definitions for aie.shim_dma_allocation operations.

This script identifies aie.shim_dma_allocation ops whose reference tile is not 
defined anywhere and adds a tile definition above them.

The aie.shim_dma_allocation operation syntax changed to require a tile reference:
  Old: aie.shim_dma_allocation @toMem (S2MM, 0, 2)
  New: aie.shim_dma_allocation @toMem (%tile_2_0, S2MM, 0)

This script automatically adds the missing tile definitions:
  %tile_2_0 = aie.tile(2, 0)
  aie.shim_dma_allocation @toMem (%tile_2_0, S2MM, 0)

Usage:
  python3 add_missing_tile_defs.py <file_or_directory>

Examples:
  python3 add_missing_tile_defs.py test/Conversion/DmaToNpu/dma_to_npu.mlir
  python3 add_missing_tile_defs.py test/
"""

import re
import sys
from pathlib import Path


def extract_tile_ref(line):
    """Extract tile reference from aie.shim_dma_allocation line.
    
    Example: aie.shim_dma_allocation @toMem (%tile_0_0, S2MM, 0)
    Returns: 'tile_0_0' (without the %)
    """
    match = re.search(r'aie\.shim_dma_allocation\s+\S+\s+\(%tile_(\d+)_(\d+)', line)
    if match:
        col, row = match.group(1), match.group(2)
        return f'tile_{col}_{row}'
    return None


def extract_tile_coords(tile_name):
    """Extract column and row from tile name.
    
    Example: 'tile_0_0' -> (0, 0)
             'tile_1_0' -> (1, 0)
    """
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def find_tile_definitions(content):
    """Find all tile definitions in the content.
    
    Returns a set of tile names that are defined (e.g., {'tile_0_0', 'tile_1_0'})
    """
    defined_tiles = set()
    # Pattern: %tile_X_Y = aie.tile(X, Y)
    pattern = r'%tile_(\d+)_(\d+)\s*=\s*aie\.tile\('
    for match in re.finditer(pattern, content):
        col, row = match.group(1), match.group(2)
        defined_tiles.add(f'tile_{col}_{row}')
    return defined_tiles


def process_file(filepath):
    """Process a single MLIR file to add missing tile definitions."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split content into lines for processing
    lines = content.split('\n')
    modified = False
    new_lines = []
    
    # Track defined tiles per module (reset on "// -----" or at start of aie.device)
    defined_tiles_in_module = set()
    inside_aie_device = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Track if we're inside an aie.device block
        if 'aie.device(' in line and not line.strip().startswith('//'):
            inside_aie_device = True
            defined_tiles_in_module = set()
            # Re-scan upcoming lines in this module for existing tile definitions
            for future_line in lines[i:]:
                if '// -----' in future_line:
                    break  # Stop at next separator
                match = re.match(r'\s*%tile_(\d+)_(\d+)\s*=\s*aie\.tile\(', future_line)
                if match:
                    col, row = match.group(1), match.group(2)
                    defined_tiles_in_module.add(f'tile_{col}_{row}')
        
        # Reset when we hit a module separator
        if '// -----' in line:
            inside_aie_device = False
            defined_tiles_in_module = set()
        
        # Check if this line defines a tile (only count if inside aie.device)
        if inside_aie_device and not line.strip().startswith('//'):
            match = re.match(r'\s*%tile_(\d+)_(\d+)\s*=\s*aie\.tile\(', line)
            if match:
                col, row = match.group(1), match.group(2)
                defined_tiles_in_module.add(f'tile_{col}_{row}')
        
        # Check if this is a shim_dma_allocation line (only process if inside aie.device)
        if inside_aie_device and 'aie.shim_dma_allocation' in line and not line.strip().startswith('//'):
            tile_ref = extract_tile_ref(line)
            
            if tile_ref and tile_ref not in defined_tiles_in_module:
                # Extract tile coordinates
                coords = extract_tile_coords(tile_ref)
                
                if coords:
                    col, row = coords
                    # Add the tile definition before this line
                    indent = len(line) - len(line.lstrip())
                    tile_def = ' ' * indent + f'%{tile_ref} = aie.tile({col}, {row})'
                    new_lines.append(tile_def)
                    defined_tiles_in_module.add(tile_ref)  # Mark as defined
                    modified = True
                    print(f"  Added: %{tile_ref} = aie.tile({col}, {row})")
        
        new_lines.append(line)
        i += 1
    
    if modified:
        # Write back the modified content
        with open(filepath, 'w') as f:
            f.write('\n'.join(new_lines))
        return True
    
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 add_missing_tile_defs.py <file_or_directory>")
        print("\nExamples:")
        print("  python3 add_missing_tile_defs.py test/Conversion/DmaToNpu/dma_to_npu.mlir")
        print("  python3 add_missing_tile_defs.py test/")
        sys.exit(1)
    
    target = Path(sys.argv[1])
    
    if not target.exists():
        print(f"Error: {target} does not exist")
        sys.exit(1)
    
    # Collect all .mlir files to process
    if target.is_file():
        files = [target]
    else:
        files = list(target.rglob('*.mlir'))
    
    if not files:
        print(f"No .mlir files found in {target}")
        sys.exit(0)
    
    print(f"Processing {len(files)} file(s)...\n")
    
    modified_count = 0
    for filepath in files:
        print(f"Processing: {filepath}")
        if process_file(filepath):
            modified_count += 1
            print(f"  ✓ Modified\n")
        else:
            print(f"  No changes needed\n")
    
    print(f"\nSummary: Modified {modified_count} out of {len(files)} file(s)")


if __name__ == '__main__':
    main()
