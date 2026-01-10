import os
from matplotlib.pyplot import step
from tensorboard.backend.event_processing import event_accumulator
import json
import csv


def extract_scalar_data(events_dir):
    """
    Extracts scalar data from all tfevents files in a directory.

    Args:
        events_dir (str): The directory containing the tfevents files.
    """
    # Get all tfevents files in the directory
    event_files = [
        os.path.join(events_dir, f)
        for f in os.listdir(events_dir)
        if f.startswith("events.out.tfevents")
    ]

    if not event_files:
        print(f"No tfevents files found in {events_dir}")
        return

    for event_file in event_files:
        print(f"Processing event file: {event_file}")

        # Initialize an EventAccumulator
        ea = event_accumulator.EventAccumulator(
            event_file,
            size_guidance={  # see below for reference
                event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                event_accumulator.IMAGES: 4,
                event_accumulator.AUDIO: 4,
                event_accumulator.SCALARS: 0,  # 0 means load all
                event_accumulator.HISTOGRAMS: 1,
            },
        )

        # Load the events from the file
        ea.Reload()

        # Get all scalar tags
        scalar_tags = ea.Tags()["scalars"]

        if not scalar_tags:
            print("  No scalar data found in this file.")
            continue

        # Extract and print values for specific tags
        losses = {}
        for tag in ["train/l1loss", "train/ssimloss", "train/num_GS"]:
            steps = []
            values = []
            if tag in scalar_tags:
                scalars = ea.Scalars(tag)
                for scalar in scalars:
                    steps.append(scalar.step)
                    values.append(scalar.value)
            else:
                print(f"  Tag: {tag} not found in this file.")
            losses[tag] = (steps, values)

        return losses


if __name__ == "__main__":
    # Directory containing the TensorBoard event files
    # You can change this to the directory you are interested in
    experiment_dirs = {
        "gs_rgb_default": "/home/minhtran/Code/data/vocap/minh_2/gsplat_results/frames/00000001/rgb/default/tb",
        "gs_rgb_mcmc": "/home/minhtran/Code/data/vocap/minh_2/gsplat_results/frames/00000001/rgb/mcmc/tb",
        "gs_rgba_default": "/home/minhtran/Code/data/vocap/minh_2/gsplat_results/frames/00000001/rgba/default_random_bkgd/tb",
        "gs_rgba_mcmc": "/home/minhtran/Code/data/vocap/minh_2/gsplat_results/frames/00000001/rgba/mcmc_random_bkgd/tb",
    }

    all_losses = {}

    for exp_name, log_dir in experiment_dirs.items():
        print(f"Extracting data for experiment: {exp_name}")
        losses = extract_scalar_data(log_dir)
        print(f"Extracted losses for {exp_name}: {losses}")
        all_losses[exp_name] = losses

    # Save as JSON
    json_file = "losses_data.json"
    with open(json_file, "w") as f:
        # Convert to serializable format
        json_data = {}
        for exp_name, losses in all_losses.items():
            json_data[exp_name] = {
                tag: {"steps": steps, "values": values}
                for tag, (steps, values) in losses.items()
            }
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON to {json_file}")

    # Save as CSV
    csv_file = "losses_data.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "metric", "step", "value"])
        for exp_name, losses in all_losses.items():
            for tag, (steps, values) in losses.items():
                for step, value in zip(steps, values):
                    writer.writerow([exp_name, tag, step, value])
    print(f"Saved CSV to {csv_file}")
