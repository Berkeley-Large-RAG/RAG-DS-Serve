import os
import sys
import subprocess
import click
from .data_utils import prepare_for_upload, prepare_after_download, check_and_prepare_index

@click.group()
def cli():
    """Massive Serve CLI"""
    pass

@cli.command(name='upload-data')
@click.option('--chunk_size_gb', type=float, default=40, help='Maximum size of each chunk in GB')
@click.option('--domain_name', type=str, default='dpr_wiki_contriever', help='Domain name')
def upload_data(chunk_size_gb, domain_name):
    """Upload data to Hugging Face, automatically splitting large files"""
    # Set datastore path to `~` if it is not already set
    env = os.environ.copy()
    if 'DATASTORE_PATH' not in env:
        env['DATASTORE_PATH'] = os.path.expanduser('~')
    
    data_dir = os.path.join(os.path.expanduser(env['DATASTORE_PATH']), domain_name)
    
    # Split large files if necessary
    split_files = prepare_for_upload(data_dir, chunk_size_gb)
    if split_files:
        print(f"Split {len(split_files)} files: {split_files}")
    
    # Upload to Hugging Face
    subprocess.run(['huggingface-cli', 'upload', f'rulins/massive_serve_{domain_name}', data_dir, '--repo-type', 'dataset'])

@cli.command(name='serve')
@click.option('--domain_name', type=str, default='dpr_wiki_contriever', help='Domain name')
def serve(domain_name):
    """Run the worker node"""
    # Set the domain name
    os.environ['MASSIVE_SERVE_DOMAIN_NAME'] = domain_name
    
    # Set datastore path to `~` if it is not already set
    if 'DATASTORE_PATH' not in os.environ:
        datastore_path = input(f"Please enter a path to save the downloaded index (default: {os.path.expanduser('~')}): ").strip()
        os.environ['DATASTORE_PATH'] = datastore_path if datastore_path else os.path.expanduser('~')
    
    # Add package root to PYTHONPATH
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    
    # Check and prepare index
    save_path = os.path.join(os.path.expanduser(os.environ['DATASTORE_PATH']), domain_name)
    if not check_and_prepare_index(save_path, domain_name):
        raise FileNotFoundError("Failed to prepare index")
    
    print(f"Starting {domain_name} server...")
    # Run the worker node script
    from .api.serve import main as serve_main
    serve_main()

if __name__ == '__main__':
    cli() 