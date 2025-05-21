import zipfile
import shutil
import tempfile
from pathlib import Path

def get_available_profiles():
    """Get list of available profiles."""
    profiles_path = Path("profiles")
    return [d.name for d in profiles_path.iterdir() if d.is_dir() and d.name != "__pycache__"]

def get_available_configs(profile):
    """Get list of available configs for a profile."""
    configs_path = Path("profiles") / profile / "configs"
    return [c.stem for c in configs_path.iterdir() 
            if c.name != "__pycache__" and c.suffix == '.json']

def select_profile(profile=None):
    """Select a profile, either from argument or via prompt."""
    profiles = get_available_profiles()
    
    if len(profiles) == 0:
        print("No profiles found. Please create a profile first.")
        return None

    if profile is None:
        print("Available profiles:")
        for i, p in enumerate(profiles, 1):
            print(f"{i}. {p}")
        
        try:
            choice = int(input("\nSelect profile number (or 0 to cancel): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(profiles):
                profile = profiles[choice-1]
            else:
                print("Invalid selection.")
                return None
        except ValueError:
            print("Please enter a valid number.")
            return None
    elif profile not in profiles:
        print(f"Error: Profile '{profile}' not found.")
        return None

    return profile

def select_configs(profile):
    """Let the user select which configs to include in the export."""
    configs = get_available_configs(profile)
    
    if len(configs) == 0:
        print("No configs found for this profile.")
        return []

    selected_configs = []
    
    print("\nAvailable configs:")
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config}")
    
    print("\nSelect configs to export (comma-separated numbers, or 'all' for all configs, or 0 to cancel):")
    choice = input("> ").strip().lower()
    
    if choice == "0":
        return []
    
    if choice == "all":
        return configs
    
    try:
        selections = [int(x.strip()) for x in choice.split(",") if x.strip()]
        for selection in selections:
            if 1 <= selection <= len(configs):
                selected_configs.append(configs[selection-1])
            else:
                print(f"Ignoring invalid selection: {selection}")
    except ValueError:
        print("Invalid input. Please enter comma-separated numbers.")
        return []
    return selected_configs

def export_profile(profile, configs=None, output_path=None):
    """
    Export a profile and its outputs as a zip file.
    
    Args:
        profile (str): Name of the profile to export
        configs (list): List of configs to include (if None, user will be prompted)
        output_path (str): Path where to save the zip file (default: current directory)
    
    Returns:
        str: Path to the created zip file or None if export failed
    """
    profile_path = Path("profiles") / profile
    if not profile_path.exists():
        print(f"Error: Profile '{profile}' not found.")
        return None
    
    if configs is None:
        configs = select_configs(profile)
        
    if not configs:
        print("No configs selected for export.")
        return None
    
    # Create temp directory for organizing files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        # Create directory structure
        profile_export_path = temp_dir_path / "profiles" / profile
        profile_export_path.mkdir(parents=True, exist_ok=True)
        
        # Copy profile code
        for item in profile_path.iterdir():
            # Skip __pycache__ directories
            if item.name == "__pycache__":
                continue
                
            if item.is_dir():
                # Copy the directory excluding __pycache__
                dst = profile_export_path / item.name
                shutil.copytree(item, dst, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(item, profile_export_path / item.name)
        
        # Copy output folders for selected configs
        for config in configs:
            out_dir = Path("out") / profile / config
            if out_dir.exists():
                dst_dir = temp_dir_path / "out" / profile / config
                dst_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy output files and directories
                for item in out_dir.iterdir():
                    # Skip __pycache__ directories
                    if item.name == "__pycache__":
                        continue
                        
                    dst = dst_dir / item.name
                    if item.is_dir():
                        # Copy the directory excluding __pycache__
                        shutil.copytree(item, dst, ignore=shutil.ignore_patterns("__pycache__"))
                    else:
                        shutil.copy2(item, dst)
        
        # Create zip file
        zip_filename = f"{profile}.zip" if output_path is None else Path(output_path) / f"{profile}.zip"
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in Path(temp_dir).rglob('**/*'):
                # Skip __pycache__ directories during zip creation
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(temp_dir_path)
                    zipf.write(file_path, arcname)
    
    print(f"Profile '{profile}' exported successfully to {zip_filename}")
    return str(zip_filename)

def export_profile_interactive():
    """Interactive function to export a profile."""
    profile = select_profile()
    if profile is None:
        return
    
    configs = select_configs(profile)
    if not configs:
        return
    
    return export_profile(profile, configs)

def main():
    """Main function to run the exporter from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export AlifeHub profiles as zip files')
    parser.add_argument('--profile', '-p', type=str, help='Profile name to export')
    parser.add_argument('--configs', '-c', type=str, nargs='+', help='Configs to include (if omitted, user will be prompted)')
    parser.add_argument('--output', '-o', type=str, help='Output directory for the zip file')
    
    args = parser.parse_args()
    
    if args.profile or args.configs or args.output:
        # Command line mode
        profile = select_profile(args.profile)
        if profile is None:
            return
        
        configs = args.configs if args.configs else None
        
        export_profile(profile, configs, args.output)
    else:
        # Interactive mode
        export_profile_interactive()

if __name__ == "__main__":
    main() 