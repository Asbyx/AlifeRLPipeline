import os
import zipfile
import shutil
import tempfile
import time

def get_available_profiles():
    """Get list of available profiles."""
    return [d for d in os.listdir("profiles") if os.path.isdir(os.path.join("profiles", d)) and d != "__pycache__"]

def get_available_configs(profile):
    """Get list of available configs for a profile."""
    return [c.split('.')[0] for c in os.listdir(os.path.join("profiles", profile, "configs")) 
            if c != "__pycache__" and c.endswith('.json')]

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
    if not os.path.exists(os.path.join("profiles", profile)):
        print(f"Error: Profile '{profile}' not found.")
        return None
    
    if configs is None:
        configs = select_configs(profile)
        
    if not configs:
        print("No configs selected for export.")
        return None
    
    # Create temp directory for organizing files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directory structure
        os.makedirs(os.path.join(temp_dir, f"profiles/{profile}"), exist_ok=True)
        
        # Copy profile code
        profile_dir = os.path.join("profiles", profile)
        for item in os.listdir(profile_dir):
            src = os.path.join(profile_dir, item)
            dst = os.path.join(temp_dir, f"profiles/{profile}/{item}")
            
            # Skip __pycache__ directories
            if os.path.basename(src) == "__pycache__":
                continue
                
            if os.path.isdir(src):
                # Copy the directory excluding __pycache__
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(src, dst)
        
        # Copy output folders for selected configs
        for config in configs:
            out_dir = os.path.join("out", profile, config)
            if os.path.exists(out_dir):
                dst_dir = os.path.join(temp_dir, f"out/{profile}/{config}")
                os.makedirs(dst_dir, exist_ok=True)
                
                # Copy output files and directories
                for item in os.listdir(out_dir):
                    src = os.path.join(out_dir, item)
                    dst = os.path.join(dst_dir, item)
                    
                    # Skip __pycache__ directories
                    if os.path.basename(src) == "__pycache__":
                        continue
                        
                    if os.path.isdir(src):
                        # Copy the directory excluding __pycache__
                        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
                    else:
                        shutil.copy2(src, dst)
        
        # Create zip file
        zip_filename = f"{profile}.zip" if output_path is None else os.path.join(output_path, f"{profile}.zip")
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                # Skip __pycache__ directories during zip creation
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
    
    print(f"Profile '{profile}' exported successfully to {zip_filename}")
    return zip_filename

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