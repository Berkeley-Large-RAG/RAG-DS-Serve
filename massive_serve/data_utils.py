import os
import json
import shutil
import faiss
from tqdm import tqdm


def get_file_size_gb(file_path):
    """Get file size in GB"""
    return os.path.getsize(file_path) / (1024 * 1024 * 1024)

def split_large_file(file_path, output_dir, chunk_size_gb=40):
    """
    Split a large file into smaller chunks.
    For FAISS files, use special handling.
    For other files, use simple binary splitting.
    """
    file_size_gb = get_file_size_gb(file_path)
    if file_size_gb <= chunk_size_gb:
        return False  # No need to split
    
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(file_path)
    
    # Special handling for FAISS files
    if file_path.endswith('.faiss'):
        return split_faiss_index(file_path, output_dir, chunk_size_gb)
    
    # For other files, use binary splitting
    chunk_size = int(chunk_size_gb * 1024 * 1024 * 1024)  # Convert to bytes
    with open(file_path, 'rb') as f:
        chunk_num = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            chunk_path = os.path.join(output_dir, f"{file_name}.chunk_{chunk_num:03d}")
            with open(chunk_path, 'wb') as chunk_file:
                chunk_file.write(chunk)
            chunk_num += 1
    
    # Save metadata
    metadata = {
        'original_file': file_name,
        'n_chunks': chunk_num,
        'chunk_size_gb': chunk_size_gb,
        'total_size_gb': file_size_gb
    }
    
    with open(os.path.join(output_dir, f"{file_name}.metadata.json"), 'w') as f:
        json.dump(metadata, f)
    
    return True

def combine_large_file(input_dir, output_path):
    """
    Combine split file chunks back into a single file.
    For FAISS files, use special handling.
    For other files, use simple binary combining.
    """
    try:
        print(f"Looking for metadata in {input_dir}")
        # Find metadata file - check both naming patterns
        metadata_files = [f for f in os.listdir(input_dir) if f.endswith('.metadata.json') or f == 'index_metadata.json']
        if not metadata_files:
            print(f"No metadata files found in {input_dir}")
            return False
        
        metadata_path = os.path.join(input_dir, metadata_files[0])
        print(f"Found metadata file: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # For FAISS files, the metadata might not have original_file
        if 'original_file' not in metadata:
            print("No original_file in metadata, assuming FAISS index")
            return combine_faiss_index(input_dir, output_path)
            
        original_file = metadata['original_file']
        print(f"Original file name: {original_file}")
        
        # Special handling for FAISS files
        if original_file.endswith('.faiss'):
            print("Detected FAISS file, using special handling")
            return combine_faiss_index(input_dir, output_path)
        
        # For other files, use binary combining
        print(f"Combining {metadata['n_chunks']} chunks into {output_path}")
        with open(output_path, 'wb') as outfile:
            for i in range(metadata['n_chunks']):
                chunk_path = os.path.join(input_dir, f"{original_file}.chunk_{i:03d}")
                print(f"Processing chunk {i+1}/{metadata['n_chunks']}: {chunk_path}")
                if not os.path.exists(chunk_path):
                    print(f"Error: Chunk file not found: {chunk_path}")
                    return False
                with open(chunk_path, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)
        
        print(f"Successfully combined file: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error combining file: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def split_faiss_index(input_path, output_dir, chunk_size_gb=40):
    """Split a large FAISS index into smaller chunks."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading index from {input_path}")
    index = faiss.read_index(input_path)
    
    n_vectors = index.ntotal
    dim = index.d
    
    bytes_per_vector = dim * 4
    vectors_per_chunk = int((chunk_size_gb * 1024 * 1024 * 1024) / bytes_per_vector)
    n_chunks = (n_vectors + vectors_per_chunk - 1) // vectors_per_chunk
    
    print(f"Splitting index into {n_chunks} chunks")
    
    for i in tqdm(range(n_chunks)):
        start_idx = i * vectors_per_chunk
        end_idx = min((i + 1) * vectors_per_chunk, n_vectors)
        
        chunk_index = faiss.IndexFlatL2(dim)
        vectors = index.reconstruct_n(start_idx, end_idx - start_idx)
        chunk_index.add(vectors)
        
        chunk_path = os.path.join(output_dir, f"index_chunk_{i:03d}.faiss")
        faiss.write_index(chunk_index, chunk_path)
    
    metadata = {
        'n_chunks': n_chunks,
        'dim': dim,
        'total_vectors': n_vectors,
        'vectors_per_chunk': vectors_per_chunk,
        'index_type': type(index).__name__
    }
    
    with open(os.path.join(output_dir, 'index_metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    return True

def combine_faiss_index(input_dir, output_path):
    """Combine split FAISS index chunks back into a single index."""
    try:
        print(f"Looking for FAISS metadata in {input_dir}")
        metadata_path = os.path.join(input_dir, 'index_metadata.json')
        if not os.path.exists(metadata_path):
            print(f"FAISS metadata file not found: {metadata_path}")
            return False
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        n_chunks = metadata['n_chunks']
        dim = metadata['dim']
        
        print(f"Combining {n_chunks} FAISS chunks into a single index")
        print(f"Index dimension: {dim}")
        
        combined_index = faiss.IndexFlatL2(dim)
        
        for i in tqdm(range(n_chunks)):
            chunk_path = os.path.join(input_dir, f"index_chunk_{i:03d}.faiss")
            print(f"Processing chunk {i+1}/{n_chunks}: {chunk_path}")
            if not os.path.exists(chunk_path):
                print(f"Error: Chunk file not found: {chunk_path}")
                return False
                
            chunk_index = faiss.read_index(chunk_path)
            vectors = chunk_index.reconstruct_n(0, chunk_index.ntotal)
            combined_index.add(vectors)
        
        print(f"Saving combined index to {output_path}")
        faiss.write_index(combined_index, output_path)
        print("Successfully combined FAISS index")
        return True
        
    except Exception as e:
        print(f"Error combining FAISS index: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def prepare_for_upload(data_dir, chunk_size_gb=40):
    """
    Prepare all large files in the data directory for upload by splitting them if necessary.
    Returns a list of files that were split.
    """
    split_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if get_file_size_gb(file_path) > chunk_size_gb:
                rel_path = os.path.relpath(file_path, data_dir)
                output_dir = os.path.join(data_dir, f"{rel_path}.split")
                if split_large_file(file_path, output_dir, chunk_size_gb):
                    split_files.append(rel_path)
                    # Remove original file after successful split
                    os.remove(file_path)
    return split_files

def prepare_after_download(data_dir):
    """
    After downloading, combine any split files back into their original form.
    """
    print(f"Scanning {data_dir} for split files...")
    for root, dirs, _ in os.walk(data_dir):
        for dir_name in dirs:
            if dir_name.endswith('.split'):
                split_dir = os.path.join(root, dir_name)
                original_file = dir_name[:-6]  # Remove .split suffix
                output_path = os.path.join(root, original_file)
                print(f"Combining split files from {split_dir} into {output_path}")
                if combine_large_file(split_dir, output_path):
                    print(f"Successfully combined {original_file}")
                    # Remove split directory after successful combination
                    shutil.rmtree(split_dir)
                else:
                    print(f"Warning: Failed to combine {original_file}")
    
    # Special handling for index directory
    index_dir = os.path.join(data_dir, 'index')
    if os.path.exists(index_dir):
        print(f"Checking index directory: {index_dir}")
        index_files = [f for f in os.listdir(index_dir) if f.endswith('.faiss')]
        if not index_files:
            print("No FAISS index found, checking for split files...")
            split_dirs = [d for d in os.listdir(index_dir) if d.endswith('.split')]
            for split_dir in split_dirs:
                split_path = os.path.join(index_dir, split_dir)
                print(f"\nContents of split directory {split_path}:")
                try:
                    files = os.listdir(split_path)
                    print("Files found:")
                    for f in files:
                        file_path = os.path.join(split_path, f)
                        size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                        print(f"  - {f} ({size:.2f} MB)")
                except Exception as e:
                    print(f"Error listing directory contents: {str(e)}")
                
                output_path = os.path.join(index_dir, 'index_IVFFlat.100000.768.2048.faiss')
                print(f"Combining index from {split_path} into {output_path}")
                if combine_large_file(split_path, output_path):
                    print("Successfully combined index")
                    shutil.rmtree(split_path)
                else:
                    print("Warning: Failed to combine index")

def check_and_prepare_index(save_path, domain_name):
    """
    Check if index exists (either merged or split) and download/combine if needed.
    Returns True if index exists or was successfully downloaded and prepared.
    """
    index_dir = os.path.join(save_path, 'index')
    
    # Check for merged index
    index_exists = False
    if os.path.exists(index_dir):
        index_files = [f for f in os.listdir(index_dir) if f.endswith('.faiss')]
        if len(index_files) == 1:
            index_path = os.path.join(index_dir, index_files[0])
            if os.path.exists(index_path):
                print(f"Found existing merged index at {index_path}")
                index_exists = True
    
    # Check for split index if merged index not found
    if not index_exists and os.path.exists(index_dir):
        split_dirs = [d for d in os.listdir(index_dir) if d.endswith('.split')]
        if split_dirs:
            print(f"Found split index directories: {split_dirs}")
            index_exists = True
    
    # Download only if neither version exists
    if not index_exists:
        print("No existing index found, downloading from Hugging Face...")
        import subprocess
        subprocess.run(['huggingface-cli', 'download', f'rulins/massive_serve_{domain_name}', '--repo-type', 'dataset', '--local-dir', save_path])
        
        # Combine any split files
        print("Combining split files...")
        prepare_after_download(save_path)
    else:
        print("Using existing index files")
    
    # Verify that the index file exists
    if not os.path.exists(index_dir):
        return False
        
    index_files = [f for f in os.listdir(index_dir) if f.endswith('.faiss')]
    if len(index_files) != 1:
        raise FileNotFoundError(f"Expected exactly one .faiss file in {index_dir}, found {len(index_files)}")
    index_path = os.path.join(index_dir, index_files[0])
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found at {index_path} after combining split files")
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Handle large files for Hugging Face upload/download')
    parser.add_argument('--mode', choices=['split', 'combine'], required=True, help='Mode: split for upload or combine after download')
    parser.add_argument('--data_dir', required=True, help='Directory containing the data files')
    parser.add_argument('--chunk_size_gb', type=float, default=40, help='Maximum size of each chunk in GB')
    
    args = parser.parse_args()
    
    if args.mode == 'split':
        split_files = prepare_for_upload(args.data_dir, args.chunk_size_gb)
        print(f"Split {len(split_files)} files: {split_files}")
    else:
        prepare_after_download(args.data_dir)
        print("Combined all split files") 