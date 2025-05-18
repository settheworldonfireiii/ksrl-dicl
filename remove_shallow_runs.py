import os
import re
import shutil # For deleting folders (use with caution!)

def find_latest_checkpoint(model_dir: str, prefix: str, suffix: str):
    """
    Finds the checkpoint file with the highest iteration number in its name.
    Assumes filename format: <prefix><number><suffix>
    Returns:
        A tuple (filepath, step_number) or (None, None) if not found.
    """
    latest_step = -1
    latest_checkpoint_file_name = None
    
    pattern_str = f"^{re.escape(prefix)}(\\d+){re.escape(suffix)}$"
    try:
        pattern = re.compile(pattern_str)
    except re.error as e:
        print(f"  Error compiling regex with prefix='{prefix}', suffix='{suffix}': {e}")
        return None, None

    try:
        if not os.path.isdir(model_dir): # Check if model_dir exists before listing
            print(f"  Warning: Directory not found for checkpoint search: {model_dir}")
            return None, None
        for filename in os.listdir(model_dir):
            match = pattern.match(filename)
            if match:
                try:
                    step_number = int(match.group(1))
                    # if step_number <= 40000: 
                    #     continue
                    if step_number > latest_step:
                        latest_step = step_number
                        latest_checkpoint_file_name = filename
                except ValueError:
                    print(f"  Warning: Could not parse step number from {filename} in {model_dir}")
                    continue
    except FileNotFoundError: 
        print(f"  Warning: Directory not found during checkpoint search: {model_dir}")
        return None, None
    except Exception as e:
        print(f"  Warning: Error listing files in {model_dir} for checkpoints: {e}")
        return None, None

    if latest_checkpoint_file_name:
        return os.path.join(model_dir, latest_checkpoint_file_name), latest_step
    else:
        return None, None

def delete_folders_based_on_checkpoints(
    base_scan_directory: str, 
    checkpoint_prefix: str, 
    checkpoint_suffix: str, 
    deletion_step_threshold: int, # Folders with latest checkpoint step > this will be deleted
    dry_run: bool = True
):
    """
    Scans subdirectories in 'base_scan_directory'.
    Deletes a subdirectory if:
    1. Its latest checkpoint step is GREATER THAN deletion_step_threshold.
    2. No checkpoint is found in the subdirectory.
    """
    if not os.path.isdir(base_scan_directory):
        print(f"Error: Base scan directory '{base_scan_directory}' not found.")
        return

    print(f"\n--- Starting scan in '{base_scan_directory}' ---")
    if dry_run:
        print("--- OPERATING IN DRY RUN MODE: No folders will be deleted. ---")
    else:
        print("--- !!! LIVE RUN MODE: Folders WILL be deleted. !!! ---")

    folders_to_delete_count = 0
    folders_scanned_count = 0

    for item_name in os.listdir(base_scan_directory):
        item_path = os.path.join(base_scan_directory, item_name)
        
        if os.path.isdir(item_path):
            folders_scanned_count += 1
            current_model_dir = item_path
            print(f"\nProcessing directory: {current_model_dir}")

            latest_ckpt_file, latest_step = find_latest_checkpoint(
                model_dir=current_model_dir,
                prefix=checkpoint_prefix,
                suffix=checkpoint_suffix
            )

            should_delete = False
            reason_for_deletion = ""

            if latest_ckpt_file and latest_step != -1:
                print(f"  Found latest checkpoint: {os.path.basename(latest_ckpt_file)} (Step: {latest_step})")
                if latest_step <= deletion_step_threshold:
                    should_delete = True
                    reason_for_deletion = f"Latest step {latest_step} <= threshold {deletion_step_threshold}"
                else:
                    print(f"  Checkpoint step {latest_step} is not <= threshold {deletion_step_threshold}. Folder kept")
            else: # No checkpoint found
                print(f"  No checkpoints found matching pattern in {current_model_dir}.")
                should_delete = True # New rule: delete if no checkpoint found
                reason_for_deletion = "No checkpoints found"

            if should_delete:
                print(f"  DECISION: Delete folder. Reason: {reason_for_deletion}.")
                folders_to_delete_count +=1
                if dry_run:
                    print(f"  [DRY RUN] Would delete folder: {current_model_dir}")
                else:
                    print(f"  DELETING FOLDER: {current_model_dir}")
                    shutil.rmtree(current_model_dir)
                
    print("\n--- Scan Complete ---")
    print(f"Total folders scanned: {folders_scanned_count}")
    if dry_run:
        print(f"Folders that would be deleted: {folders_to_delete_count}")
    else:
        print(f"Folders deleted: {folders_to_delete_count}")

# --- Configuration and Execution ---
if __name__ == "__main__":
    BASE_SCAN_DIRECTORY = "/scratch.global/lee02328/ksrl-dicl/runs" 
    CHECKPOINT_PREFIX = "actor_checkpoint_" 
    CHECKPOINT_SUFFIX = ".pth"      
    
    DELETION_THRESHOLD = 20000 

    PERFORM_DRY_RUN = True 


    if not PERFORM_DRY_RUN:
        confirmation = input(
            f"WARNING: You are about to run in LIVE MODE.\n"
            f"Folders in '{BASE_SCAN_DIRECTORY}' will be PERMANENTLY DELETED if:\n"
            f"1. Their latest checkpoint step is > {DELETION_THRESHOLD}.\n"
            f"2. No checkpoints are found in them.\n"
            f"Are you absolutely sure? Type 'YES' to proceed: "
        )
        if confirmation != "YES":
            print("Deletion cancelled by user.")
            exit()
        
    delete_folders_based_on_checkpoints(
        base_scan_directory=BASE_SCAN_DIRECTORY,
        checkpoint_prefix=CHECKPOINT_PREFIX,
        checkpoint_suffix=CHECKPOINT_SUFFIX,
        deletion_step_threshold=DELETION_THRESHOLD,
        dry_run=PERFORM_DRY_RUN
    )