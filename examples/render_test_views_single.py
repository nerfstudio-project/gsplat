from render_test_views_multiprocessing import process_single_experiment


def main():
    task_infos = [
        ("rgb", "default", "00000008"),
        ("rgb", "mcmc", "00000008"),
        ("rgba", "default_random_bkgd", "00000008"),
        ("rgba", "default_alpha_loss", "00000008"),
        ("rgba", "mcmc_random_bkgd", "00000008"),
        ("rgba", "mcmc_alpha_loss", "00000008"),
        ("rgb", "default", "00000009"),
        ("rgb", "mcmc", "00000009"),
        ("rgba", "default_random_bkgd", "00000009"),
        ("rgba", "default_alpha_loss", "00000009"),
        ("rgba", "mcmc_random_bkgd", "00000009"),
        ("rgba", "mcmc_alpha_loss", "00000009"),
    ]

    successful_runs = []
    failed_runs = []
    failed_runs_with_errors = []

    for task in task_infos:
        run_info, error_message, success = process_single_experiment(task)

        if success:
            successful_runs.append(run_info)
        else:
            failed_runs.append(run_info)
            # Store the error info for this failed run
            failed_runs_with_errors.append((run_info, error_message))

    # Save failed runs to text file
    if failed_runs:
        failed_runs_file = "failed_runs_single.txt"
        with open(failed_runs_file, "w") as f:
            f.write("FAILED RUNS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            for failed_run in failed_runs_with_errors:
                f.write(f"- {failed_run[0]}: {failed_run[1]}\n")
        print(f"\nFailed runs saved to: {failed_runs_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total successful runs: {len(successful_runs)}")
    print(f"Total failed runs: {len(failed_runs)}")
    print(f"Total failed runs with errors: {len(failed_runs_with_errors)}")

    if failed_runs:
        print("\nFAILED RUNS:")
        for failed_run in failed_runs:
            print(f"  - {failed_run}")

        # Exit with error code if there were failures
        print(f"\nERROR: {len(failed_runs)} runs failed. See details above.")
        exit(1)
    else:
        print("\nAll runs completed successfully!")


if __name__ == "__main__":
    main()
